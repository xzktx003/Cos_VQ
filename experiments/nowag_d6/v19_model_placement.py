"""Optional resident multi-GPU placement for Llama/Qwen evaluation only."""

from __future__ import annotations

import torch
from accelerate import dispatch_model
from accelerate.utils.modeling import check_device_map, find_tied_parameters

from experiments.nowag_d6.logical_state import reconstruct_logical_state


def _mapped_device(name, device_map):
    matches = [prefix for prefix in device_map if name == prefix or name.startswith(prefix + ".")]
    if not matches:
        raise ValueError(f"Device map does not cover {name}")
    return torch.device(device_map[max(matches, key=len)])


def build_multi_gpu_map(model, device_count, *, cpu_offload_layers=0):
    """Assign complete decoder blocks without overlapping parent module entries.

    A parent 'model' entry would initially move its entire subtree to one GPU
    during dispatch and can cause an otherwise avoidable out-of-memory error.
    This map covers embeddings, individual blocks, final norm, head, and other
    independent child modules (such as the shared rotary embedding) explicitly.
    """
    body = getattr(model, "model", None)
    if body is None or any(not hasattr(body, name) for name in ("embed_tokens", "layers", "norm")) or not hasattr(model, "lm_head"):
        raise ValueError("Two-GPU evaluation requires the Llama/Qwen decoder module layout")
    if not isinstance(device_count, int) or device_count < 2:
        raise ValueError("Multi-GPU evaluation requires at least two devices")
    count = len(body.layers)
    if count < device_count:
        raise ValueError("Multi-GPU evaluation requires at least one decoder layer per device")
    if not isinstance(cpu_offload_layers, int) or not 0 <= cpu_offload_layers < count:
        raise ValueError("cpu_offload_layers must be an integer in [0, decoder_layer_count)")
    resident_count = count - cpu_offload_layers
    if resident_count < device_count:
        raise ValueError("CPU offload must leave at least one decoder layer on every visible GPU")
    device_map = {"model.embed_tokens": "cuda:0"}
    device_map.update(
        {
            f"model.layers.{index}": (
                f"cuda:{min(index * device_count // resident_count, device_count - 1)}"
                if index < resident_count
                else "cpu"
            )
            for index in range(count)
        }
    )
    last_device = f"cuda:{device_count - 1}"
    device_map.update({"model.norm": last_device, "lm_head": last_device})
    for name, _ in body.named_children():
        if name not in {"embed_tokens", "layers", "norm"}:
            device_map[f"model.{name}"] = "cuda:0"
    for name, _ in model.named_children():
        if name not in {"model", "lm_head"}:
            device_map[name] = "cuda:0"
    check_device_map(model, device_map)
    for group in find_tied_parameters(model):
        destinations = {_mapped_device(name, device_map) for name in group}
        if len(destinations) > 1:
            raise ValueError(f"Tied parameters cross a multi-GPU boundary: {group}; keep this model on one GPU")
    # Includes nonpersistent buffers, which check_device_map's state_dict omits.
    for name, _ in model.named_buffers():
        _mapped_device(name, device_map)
    return device_map


def build_two_gpu_map(model):
    """Backward-compatible two-device map used by existing evaluations/tests."""
    return build_multi_gpu_map(model, 2)


def _check_resident_placement(model, device_map):
    for iterator in (model.named_parameters(), model.named_buffers()):
        for name, value in iterator:
            expected = _mapped_device(name, device_map)
            if expected.type == "cpu":
                valid = value.device.type in {"cpu", "meta"}
            else:
                valid = value.device == expected and value.device.type == "cuda"
            if not valid:
                raise ValueError(f"Nonresident evaluation placement for {name}: {value.device}, expected {expected}")


def place_evaluation_model(model, device, *, split_gpus=False, cpu_offload_layers=0):
    """Preserve normal .to(device), or dispatch a CPU model across visible GPUs.

    CUDA ordinals are relative to CUDA_VISIBLE_DEVICES. Tied weights crossing
    the requested boundary are rejected; this helper never unties or changes
    precision. Explicit trailing-layer CPU offload uses Accelerate hooks and
    executes those layers on the primary GPU. Do not call model.to() after the
    split dispatch.
    """
    device = torch.device(device)
    if not split_gpus:
        if cpu_offload_layers:
            raise ValueError("CPU layer offload requires --split-gpus")
        return model.to(device)
    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_device_count < 2:
        raise ValueError("--split-gpus requires at least two visible CUDA devices")
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("--split-gpus requires --device cuda:0 (the first visible GPU)")
    for iterator in (model.parameters(), model.buffers()):
        if any(value.device.type != "cpu" for value in iterator):
            raise ValueError("Split evaluation requires a fully materialized CPU-loaded model")
    device_map = build_multi_gpu_map(
        model,
        visible_device_count,
        cpu_offload_layers=cpu_offload_layers,
    )
    model = dispatch_model(model, device_map=device_map, main_device="cuda:0",
                           offload_buffers=False, force_hooks=True,
                           skip_keys=getattr(model, "_skip_keys_device_placement", None))
    model.hf_device_map = dict(device_map)
    _check_resident_placement(model, device_map)
    return model


@torch.no_grad()
def reconstruct_on_module(module, state):
    """Reconstruct directly where this module's resident dense weight lives."""
    if module.weight.device.type == "meta":
        raise ValueError("Logical reconstruction requires a resident weight, not meta/offloaded storage")
    weight = reconstruct_logical_state(state, module.weight.device)
    module.weight.copy_(weight.to(module.weight.dtype))
    return weight


def placement_summary(model):
    """Small placement evidence for scorer/evaluator result JSON."""
    sizes = {}
    for parameter in model.parameters():
        device = str(parameter.device)
        sizes[device] = sizes.get(device, 0) + parameter.numel() * parameter.element_size()
    device_map = getattr(model, "hf_device_map", None)
    return {"hf_device_map": dict(device_map) if device_map is not None else None,
            "parameter_bytes_by_device": sizes,
            "input_device": str(model.model.embed_tokens.weight.device),
            "output_head_device": str(model.lm_head.weight.device)}
