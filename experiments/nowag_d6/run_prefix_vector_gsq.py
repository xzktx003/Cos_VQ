"""Run NoWag-d6 -> vector GPTQ -> Vector-GSQ prefix ablations.

The model is quantized sequentially so later Hessians see activations produced
by the already quantized prefix.  WikiText2 perplexity is evaluated at selected
prefix depths, matching the gating philosophy used by the NoWag-d1 report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import inspect
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, TypeVar

import torch
import torch.nn.functional as F
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from experiments.llama_qtip.protocol import (
    infer_model_family,
    validate_token_cache_identity,
)
from experiments.nowag_d6.logical_state import logical_state_bits, reconstruct_logical_state
from experiments.nowag_d6.diagnose_block_vector_gsq import (
    LINEAR_NAMES,
    aggregate_gradient_metrics,
    aggregate_probability_metrics,
    collect_hessian_diagonals,
    get_linear_modules,
    hard_weights_and_switch_rate,
    patched_linear_weights,
    run_to_block_output,
)
from experiments.nowag_d6.diagnose_vector_gsq_gradients import (
    LocalVectorGumbelQuantizer,
    Schedule,
    build_local_candidates,
    initialize_logits,
    normalized_vectors_and_weights,
    run_full_nowag_initialization,
    seed_everything,
    vector_codebook_aware_gptq,
)
from experiments.nowag_d6.staged_vector_gsq import (
    apply_group_scales,
    anchor_quantizer_logits_,
    collect_dense_stage_targets,
    expand_group_scales as expand_group_scales,
    hidden_stage_mse,
    joint_block_refinement_loss,
    merge_selected_weights,
    run_hidden_block_with_group_scales,
    run_hidden_block_with_row_scales,
    run_hidden_stage_output,
    select_better_joint_snapshot,
    snapshot_joint_hard_state,
    staged_phase_specs,
    validate_block_refinement_contract,
)


def _load_wiki_eval():
    """Load the local evaluator without claiming the shared ``src`` name.

    Both this project and the external NoWag checkout expose a top-level
    ``src`` package.  Importing the evaluator through ``src.evaluation`` first
    pins this project's package in ``sys.modules`` and prevents NoWag's
    ``src.quantize_compress`` from being imported later.
    """
    module_path = Path(__file__).resolve().parents[2] / "src/evaluation/wiki_eval.py"
    spec = importlib.util.spec_from_file_location("gsq_wiki_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load WikiText2 evaluator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WIKI_EVAL = _load_wiki_eval()
evaluate_ppl = _WIKI_EVAL.evaluate_ppl
get_dataset = _WIKI_EVAL.get_dataset
prepare_test_dataloader = _WIKI_EVAL.prepare_test_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu-reservation-mib",
        type=int,
        default=0,
        help=(
            "Reserve this many MiB on --device for the lifetime of the run. "
            "Layerwise CPU-offload launchers use this to make GPU ownership "
            "visible while the base model is loading on CPU."
        ),
    )
    parser.add_argument(
        "--hf-device-map",
        default="",
        help=(
            "Optional Hugging Face device_map value such as 'auto' or 'cpu'. "
            "When set, the model is loaded without a full .to(device)."
        ),
    )
    parser.add_argument(
        "--hf-max-memory",
        default="",
        help=(
            "Optional JSON dict passed to from_pretrained(max_memory=...), "
            'for example {"cuda:0":"46GiB","cuda:1":"46GiB","cpu":"220GiB"}.'
        ),
    )
    parser.add_argument(
        "--layerwise-cpu-offload",
        action="store_true",
        help=(
            "Load the base model on CPU and move only the current Transformer "
            "block to --device. This single-GPU path requires --use-hidden-cache "
            "and disables inline prefix PPL."
        ),
    )
    parser.add_argument(
        "--nowag-repo",
        default=str(Path(__file__).resolve().parents[2] / "third_party" / "NoWag"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the completed contiguous layers in --checkpoint and "
            "--output. Completed logical weights are restored and replayed "
            "through every hidden cache before the first unfinished layer."
        ),
    )
    parser.add_argument("--stage", choices=("nowag", "gptq", "gsq"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--ppl-depths", default="2,4,8")
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument("--calib-sequences", type=int, default=8)
    parser.add_argument("--calib-sequence-length", type=int, default=512)
    parser.add_argument("--gptq-token-offset", type=int, default=4224)
    parser.add_argument("--train-sequences", type=int, default=1)
    parser.add_argument("--train-sequence-length", type=int, default=128)
    parser.add_argument(
        "--train-segment-length",
        type=int,
        default=0,
        help=(
            "Split every selected training row into independent contiguous "
            "segments of this length before hidden-cache construction.  Zero "
            "keeps the original sequence shape."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-microbatch-size", type=int, default=1)
    parser.add_argument("--validation-sequences", type=int, default=0)
    parser.add_argument("--validation-token-offset", type=int, default=4096)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-segment-length",
        type=int,
        default=0,
        help=(
            "Split validation rows into independent contiguous segments.  "
            "Zero keeps the original sequence shape."
        ),
    )
    parser.add_argument("--use-hidden-cache", action="store_true")
    parser.add_argument(
        "--stream-dense-targets",
        action="store_true",
        help=(
            "Recompute dense staged teachers per training microbatch instead "
            "of retaining full training-target tensors in host memory. This "
            "preserves the objective while trading extra dense forwards for "
            "a substantially smaller CPU-memory peak."
        ),
    )
    parser.add_argument("--cache-microbatch-size", type=int, default=4)
    parser.add_argument("--vector-dim", type=int, default=6)
    parser.add_argument("--bits-per-weight", type=float, default=2.0)
    parser.add_argument("--lloyd-iters", type=int, default=100)
    parser.add_argument("--local-candidates", type=int, default=16)
    parser.add_argument("--candidate-chunk", type=int, default=4096)
    parser.add_argument("--gptq-percdamp", type=float, default=0.1)
    parser.add_argument("--gptq-row-chunk", type=int, default=256)
    parser.add_argument(
        "--gptq-candidates",
        type=int,
        default=128,
        help=(
            "Number of local NoWag codewords scored exactly per vector during "
            "Vector-GPTQ. Values at least as large as the codebook restore the "
            "full-codebook search."
        ),
    )
    parser.add_argument(
        "--gptq-concurrency",
        type=int,
        default=1,
        help=(
            "Number of same-device CUDA streams used to refine independent "
            "Linear modules in the current block. One preserves the original "
            "serial execution; values from 2 to 7 enable bounded concurrency."
        ),
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--gsq-objective",
        choices=("block-all", "staged"),
        default="block-all",
        help=(
            "Vector-GSQ optimization granularity. 'block-all' preserves the "
            "legacy seven-Linear joint objective; 'staged' trains Q, K, V/O, "
            "and MLP with structure-aware activation targets."
        ),
    )
    parser.add_argument(
        "--stage-steps-qk",
        type=int,
        default=0,
        help="Steps for each Q and K phase; zero reuses --steps.",
    )
    parser.add_argument(
        "--stage-steps-vo",
        type=int,
        default=0,
        help="Steps for the joint V/O attention phase; zero reuses --steps.",
    )
    parser.add_argument(
        "--stage-steps-mlp",
        type=int,
        default=0,
        help="Steps for the joint MLP phase; zero reuses --steps.",
    )
    parser.add_argument(
        "--staged-freeze-qk",
        action="store_true",
        help=(
            "Keep Vector-GPTQ Q/K assignments fixed and train only V/O and "
            "MLP in staged mode, matching the original GSQ LLaMA structure."
        ),
    )
    parser.add_argument(
        "--staged-joint-group-scales",
        action="store_true",
        help=(
            "Jointly optimize deployable per-row/input-group scales with "
            "assignment logits inside every active staged phase."
        ),
    )
    parser.add_argument(
        "--staged-joint-scale-lr",
        type=float,
        default=1e-3,
        help="Learning rate for group log-scales in joint staged training.",
    )
    parser.add_argument(
        "--scale-finetune-steps",
        type=int,
        default=0,
        help=(
            "Optional fixed-assignment block-output refinement steps over "
            "existing output-row normalization scales."
        ),
    )
    parser.add_argument(
        "--scale-finetune-lr",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for block-local fixed-assignment row scales. The "
            "default is large enough to produce observable BF16 changes."
        ),
    )
    parser.add_argument(
        "--scale-finetune-mode",
        choices=("row", "group"),
        default="row",
        help=(
            "Continuous fixed-assignment scale parameterization. 'row' folds "
            "one output scale into NoWag row_norm; 'group' stores one FP16 "
            "scale per output row and input group and counts it in logical bpp."
        ),
    )
    parser.add_argument(
        "--scale-group-size",
        type=int,
        default=128,
        help="Input-column group size for --scale-finetune-mode=group.",
    )
    parser.add_argument(
        "--block-polish-steps",
        type=int,
        default=0,
        help=(
            "Optional joint full-block steps after the four staged phases. "
            "This keeps staged exploration but realigns all seven Linears to "
            "the final block objective."
        ),
    )
    parser.add_argument(
        "--block-polish-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--early-layers",
        type=int,
        default=0,
        help="Use the early-layer GSQ candidate/step policy for this many layers.",
    )
    parser.add_argument("--early-local-candidates", type=int, default=0)
    parser.add_argument("--early-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument(
        "--schedule", choices=("original", "gentle"), default="original"
    )
    parser.add_argument("--half-kmeans", action="store_true")
    return parser.parse_args()


def make_schedule(name: str) -> Schedule:
    if name == "original":
        return Schedule("gsq_original", 2.0, 0.05, 100.0, 500.0)
    return Schedule("gradient_friendly_control", 2.0, 0.2, 25.0, 100.0)


_LinearJobResult = TypeVar("_LinearJobResult")


def _run_bounded_linear_jobs(
    names: List[str],
    worker: Callable[[str], _LinearJobResult],
    concurrency: int,
) -> Dict[str, _LinearJobResult]:
    """Run independent Linear jobs with bounded host-side concurrency.

    Results retain ``names`` order so checkpoint and diagnostic output stay
    stable regardless of completion order. CUDA stream ownership belongs to
    ``worker``; keeping this scheduler device-agnostic makes its concurrency
    and serial-compatibility contracts testable without a GPU.
    """
    if concurrency <= 0:
        raise ValueError("gptq concurrency must be positive")
    if concurrency == 1 or len(names) <= 1:
        return {name: worker(name) for name in names}

    max_workers = min(concurrency, len(names))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="vector-gptq",
    ) as executor:
        futures = {name: executor.submit(worker, name) for name in names}
        return {name: futures[name].result() for name in names}


def _warm_vector_gptq_cuda_linalg(device: torch.device) -> None:
    """Initialize CUDA Cholesky wrappers before concurrent worker threads.

    PyTorch lazily initializes several CUDA linalg wrappers. Letting the first
    seven GPTQ threads race through that one-time initialization can raise
    ``lazy wrapper should be called at most once``. A tiny main-stream factor
    chain pays the initialization cost once without affecting GPTQ inputs.
    """
    identity = torch.eye(2, dtype=torch.float32, device=device)
    chol = torch.linalg.cholesky(identity)
    inverse = torch.cholesky_inverse(chol)
    torch.linalg.cholesky(inverse, upper=True)
    torch.cuda.synchronize(device)


def _run_vector_gptq_jobs(
    names: List[str],
    modules,
    sources,
    hessians,
    args: argparse.Namespace,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    """Refine independent Linear assignments on bounded CUDA streams."""
    concurrency = min(int(args.gptq_concurrency), len(names))

    if concurrency == 1:

        def serial_worker(name: str):
            weight = modules[name].weight.detach().float()
            return vector_codebook_aware_gptq(
                sources[name],
                weight,
                hessians[name],
                args.vector_dim,
                percdamp=args.gptq_percdamp,
                row_chunk=args.gptq_row_chunk,
                candidate_count=args.gptq_candidates,
            )

        return _run_bounded_linear_jobs(names, serial_worker, concurrency=1)

    devices = {modules[name].weight.device for name in names}
    if len(devices) != 1:
        raise ValueError(
            "Concurrent Vector-GPTQ requires every Linear in a block to be "
            f"on one device, got {sorted(map(str, devices))}"
        )
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("gptq-concurrency > 1 requires CUDA Linear weights")

    _warm_vector_gptq_cuda_linalg(device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))

    def cuda_worker(name: str):
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            stream.wait_event(ready)
            weight = modules[name].weight.detach().float()
            result = vector_codebook_aware_gptq(
                sources[name],
                weight,
                hessians[name],
                args.vector_dim,
                percdamp=args.gptq_percdamp,
                row_chunk=args.gptq_row_chunk,
                candidate_count=args.gptq_candidates,
            )
        stream.synchronize()
        return result

    return _run_bounded_linear_jobs(names, cuda_worker, concurrency)


def split_into_independent_segments(
    token_ids: torch.Tensor,
    segment_length: int,
    split_name: str,
) -> torch.Tensor:
    """Turn long token rows into shorter independent calibration examples.

    The reshape happens before embedding and block execution, so every segment
    receives self-consistent position IDs, causal context, and dense targets.
    This is deliberately different from slicing an existing hidden-state
    cache, which would mix full-sequence targets with reset position IDs.
    """
    if segment_length == 0 or token_ids.numel() == 0:
        return token_ids
    if segment_length < 0:
        raise ValueError(f"{split_name} segment length must be non-negative")
    source_length = token_ids.shape[1]
    if segment_length > source_length:
        raise ValueError(
            f"{split_name} segment length {segment_length} exceeds source "
            f"length {source_length}"
        )
    if source_length % segment_length != 0:
        raise ValueError(
            f"{split_name} source length {source_length} must be divisible by "
            f"segment length {segment_length}"
        )
    return token_ids.reshape(-1, segment_length).contiguous()


def run_hidden_block_output(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    weights: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    block = model.model.layers[layer_index]
    position_ids = torch.arange(
        hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
    ).unsqueeze(0)
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    with patched_linear_weights(modules, weights):
        kwargs = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "use_cache": False,
        }
        if "position_embeddings" in inspect.signature(block.forward).parameters:
            kwargs["position_embeddings"] = model.model.rotary_emb(
                hidden_states, position_ids
            )
        output = block(hidden_states, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def run_experiment_block_output(
    model,
    layer_index: int,
    modules,
    inputs: torch.Tensor,
    weights: Dict[str, torch.Tensor] | None,
    inputs_are_hidden: bool,
) -> torch.Tensor:
    if inputs_are_hidden:
        return run_hidden_block_output(model, layer_index, modules, inputs, weights)
    return run_to_block_output(model, layer_index, modules, inputs, weights)


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings.weight.device
    return model.model.embed_tokens.weight.device


def get_layer_device(model: torch.nn.Module, layer_index: int) -> torch.device:
    block = model.model.layers[layer_index]
    return next(block.parameters()).device


def build_model_load_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    """Build the one authoritative model-loading policy for this run."""
    kwargs: Dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if args.layerwise_cpu_offload:
        kwargs["device_map"] = "cpu"
        kwargs["low_cpu_mem_usage"] = True
    elif args.hf_device_map:
        kwargs["device_map"] = args.hf_device_map
        kwargs["low_cpu_mem_usage"] = True
        if args.hf_max_memory:
            kwargs["max_memory"] = json.loads(args.hf_max_memory)
    return kwargs


def move_layer_for_quantization(
    model: torch.nn.Module,
    layer_index: int,
    device: torch.device | str,
    enabled: bool,
) -> None:
    """Materialize just the active block (and shared RoPE) on the quant GPU."""
    if not enabled:
        return
    model.model.layers[layer_index].to(device)
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is not None:
        rotary.to(device)


def offload_layer_after_quantization(
    model: torch.nn.Module,
    layer_index: int,
    enabled: bool,
) -> None:
    """Return a completed block to CPU so the next block gets a clean GPU."""
    if enabled:
        model.model.layers[layer_index].to("cpu")


def emit_progress(event: str, **fields: object) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def prepare_inputs_for_experiment(
    model,
    layer_index: int,
    inputs: torch.Tensor,
    inputs_are_hidden: bool,
) -> torch.Tensor:
    target_device = (
        get_layer_device(model, layer_index)
        if inputs_are_hidden
        else get_model_input_device(model)
    )
    return inputs.to(target_device)


@torch.inference_mode()
def embed_token_split(
    model,
    token_ids: torch.Tensor,
    microbatch_size: int,
    _device: torch.device,
) -> torch.Tensor:
    shape = (*token_ids.shape, model.config.hidden_size)
    hidden = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    input_device = get_model_input_device(model)
    for start in range(0, token_ids.shape[0], microbatch_size):
        end = start + microbatch_size
        ids = token_ids[start:end].to(input_device)
        output = model.model.embed_tokens(ids)
        hidden[start:end].copy_(output.to("cpu", torch.bfloat16))
    return hidden


@torch.inference_mode()
def propagate_hidden_cache_in_place(
    model,
    layer_index: int,
    modules,
    hidden: torch.Tensor,
    microbatch_size: int,
    _device: torch.device,
) -> None:
    for start in range(0, hidden.shape[0], microbatch_size):
        end = min(start + microbatch_size, hidden.shape[0])
        inputs = prepare_inputs_for_experiment(
            model,
            layer_index,
            hidden[start:end],
            inputs_are_hidden=True,
        )
        output = run_hidden_block_output(
            model, layer_index, modules, inputs, weights=None
        )
        hidden[start:end].copy_(output.to("cpu", torch.bfloat16))


@torch.inference_mode()
def collect_hessians_from_hidden(
    model,
    layer_index: int,
    modules,
    hidden: torch.Tensor,
    _device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sums = {
        name: torch.zeros(
            module.in_features,
            module.in_features,
            device=module.weight.device,
            dtype=torch.float32,
        )
        for name, module in modules.items()
    }
    counts = {name: 0 for name in modules}
    handles = []
    for name, module in modules.items():

        def make_hook(key):
            def hook(_module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                sums[key].add_(x.T @ x)
                counts[key] += x.shape[0]

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(name)))
    started = time.time()
    try:
        for index in range(hidden.shape[0]):
            inputs = prepare_inputs_for_experiment(
                model,
                layer_index,
                hidden[index : index + 1],
                inputs_are_hidden=True,
            )
            run_hidden_block_output(model, layer_index, modules, inputs, weights=None)
    finally:
        for handle in handles:
            handle.remove()
    hessians = {name: sums[name] / max(counts[name], 1) for name in modules}
    elapsed = time.time() - started
    stats = {
        name: {
            "rows": float(counts[name]),
            "diag_min": float(torch.diag(hessians[name]).min()),
            "diag_max": float(torch.diag(hessians[name]).max()),
            "diag_mean": float(torch.diag(hessians[name]).mean()),
            "shared_capture_seconds": elapsed,
            "input_mode": "hidden_cache",
        }
        for name in modules
    }
    return hessians, stats


@torch.inference_mode()
def collect_dense_block_outputs(
    model,
    layer_index: int,
    modules,
    token_ids: torch.Tensor,
    batch_size: int,
    _device: torch.device,
    inputs_are_hidden: bool,
) -> torch.Tensor:
    outputs = []
    for start in range(0, token_ids.shape[0], batch_size):
        ids = prepare_inputs_for_experiment(
            model,
            layer_index,
            token_ids[start : start + batch_size],
            inputs_are_hidden=inputs_are_hidden,
        )
        output = run_experiment_block_output(
            model,
            layer_index,
            modules,
            ids,
            weights=None,
            inputs_are_hidden=inputs_are_hidden,
        )
        outputs.append(output.detach().to("cpu", torch.bfloat16))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def dataset_block_mse(
    model,
    layer_index: int,
    modules,
    token_ids: torch.Tensor,
    dense_outputs: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    batch_size: int,
    _device: torch.device,
    inputs_are_hidden: bool,
) -> float:
    squared_error = 0.0
    elements = 0
    for start in range(0, token_ids.shape[0], batch_size):
        end = start + batch_size
        ids = prepare_inputs_for_experiment(
            model,
            layer_index,
            token_ids[start:end],
            inputs_are_hidden=inputs_are_hidden,
        )
        output = run_experiment_block_output(
            model,
            layer_index,
            modules,
            ids,
            weights,
            inputs_are_hidden=inputs_are_hidden,
        ).float()
        target = dense_outputs[start:end].to(device=output.device, dtype=torch.float32)
        squared_error += float((output - target).square().sum())
        elements += output.numel()
    return squared_error / max(elements, 1)


def build_block_quantizers(
    modules,
    sources: Dict[str, torch.nn.Module],
    hessians: Dict[str, torch.Tensor],
    assignments: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    layer_index: int,
):
    quantizers = {}
    initial_assignments = {}
    candidate_metadata = {}
    for index, name in enumerate(LINEAR_NAMES):
        module = modules[name]
        weight = module.weight.detach().float()
        source = sources[name]
        vectors, vector_weights, columns = normalized_vectors_and_weights(
            source,
            weight,
            torch.diag(hessians[name]),
            args.vector_dim,
        )
        ids, distances, metadata = build_local_candidates(
            vectors,
            vector_weights,
            source.codebook.detach().float(),
            assignments[name].long(),
            args.local_candidates,
            args.candidate_chunk,
        )
        logits = initialize_logits(
            distances,
            args.std,
            args.strength,
            args.seed + layer_index * len(LINEAR_NAMES) + index,
        )
        quantizers[name] = LocalVectorGumbelQuantizer(
            source,
            ids,
            logits,
            columns,
        ).to(weight.device)
        initial_assignments[name] = assignments[name].long()
        candidate_metadata[name] = metadata
        del weight, vectors, vector_weights, distances
    return quantizers, initial_assignments, candidate_metadata


def train_block_gsq(
    model,
    layer_index: int,
    modules,
    train_ids: torch.Tensor,
    train_dense_outputs: torch.Tensor,
    validation_ids: torch.Tensor,
    validation_dense_outputs: torch.Tensor,
    sources: Dict[str, torch.nn.Module],
    hessians: Dict[str, torch.Tensor],
    assignments: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    inputs_are_hidden: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object]]:
    # Keep the stochastic GSQ stream independent from subsequent NoWag
    # initializations so stage ablations remain paired at every layer.
    seed_everything(args.seed + 1_000_000 + layer_index)
    quantizers, initial_assignments, candidate_metadata = build_block_quantizers(
        modules,
        sources,
        hessians,
        assignments,
        args,
        layer_index,
    )
    parameters = [quantizer.quant_logits for quantizer in quantizers.values()]

    hessians.clear()
    torch.cuda.empty_cache()

    schedule = make_schedule(args.schedule)
    optimizer = Lion(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * 0.1,
    )
    initial_weights, _, _ = hard_weights_and_switch_rate(
        quantizers, initial_assignments
    )
    if validation_ids.shape[0] > 0:
        selection_ids = validation_ids
        selection_outputs = validation_dense_outputs
        selection_batch_size = args.validation_batch_size
        selection_split = "validation"
    else:
        selection_ids = train_ids
        selection_outputs = train_dense_outputs
        selection_batch_size = args.train_batch_size
        selection_split = "training"
    initial_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        selection_ids,
        selection_outputs,
        initial_weights,
        selection_batch_size,
        train_dense_outputs.device
        if train_dense_outputs.is_cuda
        else next(iter(modules.values())).weight.device,
        inputs_are_hidden,
    )
    del initial_weights
    torch.cuda.empty_cache()
    selected_validation_loss = initial_loss
    selected_assignments = {
        name: value.detach().clone() for name, value in initial_assignments.items()
    }

    steps_per_epoch = math.ceil(train_ids.shape[0] / args.train_batch_size)
    checkpoints = {0, 50, 100, 200, args.steps - 1}
    checkpoints.update(range(steps_per_epoch - 1, args.steps, steps_per_epoch))
    trace = []
    started = time.time()
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(args.seed + 2_000_000 + layer_index)
    permutation = None
    device = next(iter(modules.values())).weight.device
    for step in range(args.steps):
        t = step / max(args.steps - 1, 1)
        temperature = (
            schedule.temperature_start
            + (schedule.temperature_end - schedule.temperature_start) * t
        )
        kappa = schedule.kappa_start + (schedule.kappa_end - schedule.kappa_start) * t
        batch_slot = step % steps_per_epoch
        if batch_slot == 0:
            permutation = torch.randperm(
                train_ids.shape[0], generator=permutation_generator
            )
        start = batch_slot * args.train_batch_size
        end = min(start + args.train_batch_size, train_ids.shape[0])
        batch_indices = permutation[start:end]
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        global_batch_size = batch_indices.numel()
        for micro_start in range(0, global_batch_size, args.train_microbatch_size):
            micro_end = min(
                micro_start + args.train_microbatch_size,
                global_batch_size,
            )
            micro_indices = batch_indices[micro_start:micro_end]
            ids_batch = prepare_inputs_for_experiment(
                model,
                layer_index,
                train_ids[micro_indices],
                inputs_are_hidden=inputs_are_hidden,
            )
            soft_weights = {
                name: quantizer(temperature, kappa, add_gumbel=True)[0]
                for name, quantizer in quantizers.items()
            }
            output = run_experiment_block_output(
                model,
                layer_index,
                modules,
                ids_batch,
                soft_weights,
                inputs_are_hidden,
            )
            dense_batch = train_dense_outputs[micro_indices].to(
                device=output.device, dtype=torch.float32
            )
            micro_loss = F.mse_loss(output.float(), dense_batch)
            fraction = micro_indices.numel() / max(global_batch_size, 1)
            (micro_loss * fraction).backward()
            train_loss += float(micro_loss.detach()) * fraction
            del soft_weights, output, micro_loss, ids_batch, dense_batch
        gradient_stats = aggregate_gradient_metrics(quantizers)
        optimizer.step()
        scheduler.step()

        if step in checkpoints:
            with torch.no_grad():
                hard_weights, switch_rate, per_linear = hard_weights_and_switch_rate(
                    quantizers, initial_assignments
                )
            hard_loss = dataset_block_mse(
                model,
                layer_index,
                modules,
                selection_ids,
                selection_outputs,
                hard_weights,
                selection_batch_size,
                device,
                inputs_are_hidden,
            )
            if hard_loss < selected_validation_loss:
                selected_validation_loss = hard_loss
                selected_assignments = {
                    name: quantizer.hard_weight()[1].detach().clone()
                    for name, quantizer in quantizers.items()
                }
            trace.append(
                {
                    "step": float(step),
                    "epoch": float(step // steps_per_epoch),
                    "temperature": float(temperature),
                    "kappa": float(kappa),
                    "soft_train_batch_mse": train_loss,
                    "hard_selection_mse": float(hard_loss),
                    "selected_hard_mse": float(selected_validation_loss),
                    "assignment_switch_rate": float(switch_rate),
                    "per_linear_switch_rate": per_linear,
                    **aggregate_probability_metrics(quantizers, temperature, kappa),
                    **gradient_stats,
                }
            )

    final_weights = {}
    final_assignments = {}
    with torch.no_grad():
        for name, quantizer in quantizers.items():
            current_ids = selected_assignments[name]
            final_weights[name] = quantizer.weight_from_global_ids(current_ids)
            final_assignments[name] = current_ids
    final_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        selection_ids,
        selection_outputs,
        final_weights,
        selection_batch_size,
        device,
        inputs_are_hidden,
    )
    final_switch_rate = sum(
        float((final_assignments[name] != initial_assignments[name]).sum())
        for name in LINEAR_NAMES
    ) / max(sum(initial_assignments[name].numel() for name in LINEAR_NAMES), 1)

    return (
        final_weights,
        final_assignments,
        {
            "objective": "block_all",
            "schedule": asdict(schedule),
            "candidate_metadata": candidate_metadata,
            "selection_split": selection_split,
            "train_sequences": int(train_ids.shape[0]),
            "validation_sequences": int(validation_ids.shape[0]),
            "train_batch_size": int(args.train_batch_size),
            "train_microbatch_size": int(args.train_microbatch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "epochs_equivalent": float(args.steps / steps_per_epoch),
            "initial_hard_block_mse": float(initial_loss),
            "final_hard_block_mse": float(final_loss),
            "hard_block_mse_relative_change": float(
                (final_loss - initial_loss) / initial_loss.clamp_min(1e-30)
                if isinstance(initial_loss, torch.Tensor)
                else (final_loss - initial_loss) / max(initial_loss, 1e-30)
            ),
            "accepted_training": bool(selected_validation_loss < initial_loss),
            "final_assignment_switch_rate": float(final_switch_rate),
            "training_seconds": time.time() - started,
            "trace": trace,
        },
    )


def _staged_step_count(args: argparse.Namespace, phase_name: str) -> int:
    if phase_name in {"q_proj", "k_proj"}:
        return args.stage_steps_qk or args.steps
    if phase_name == "v_o":
        return args.stage_steps_vo or args.steps
    if phase_name == "mlp":
        return args.stage_steps_mlp or args.steps
    raise ValueError(f"Unknown staged phase: {phase_name}")


def _training_checkpoints(steps: int, steps_per_epoch: int) -> set[int]:
    checkpoints = {0, steps - 1}
    checkpoints.update(marker for marker in (50, 100, 200) if marker < steps)
    checkpoints.update(range(steps_per_epoch - 1, steps, steps_per_epoch))
    return checkpoints


@torch.no_grad()
def stage_training_target_batch(
    model,
    layer_index: int,
    modules,
    hidden_batch: torch.Tensor,
    cached_targets: torch.Tensor,
    batch_indices: torch.Tensor,
    objective: str,
    stream_dense_targets: bool,
    device: torch.device,
) -> torch.Tensor:
    """Materialize one dense staged target without changing its semantics.

    Cached mode preserves the original whole-dataset teacher path. Streaming
    mode runs the unpatched dense current block on the same quantized-prefix
    hidden batch. The target is detached in both cases; only candidate
    assignment logits and optional scales receive gradients.
    """
    if stream_dense_targets:
        target = run_hidden_stage_output(
            model,
            layer_index,
            modules,
            hidden_batch,
            None,
            objective,
        )
        return target.detach().to(device=device, dtype=torch.float32)
    if cached_targets.shape[0] == 0:
        raise ValueError("cached staged targets are empty")
    return cached_targets[batch_indices].to(
        device=device,
        dtype=torch.float32,
    )


def train_staged_block_gsq(
    model,
    layer_index: int,
    modules,
    train_ids: torch.Tensor,
    train_dense_outputs: torch.Tensor,
    validation_ids: torch.Tensor,
    validation_dense_outputs: torch.Tensor,
    sources: Dict[str, torch.nn.Module],
    hessians: Dict[str, torch.Tensor],
    assignments: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, object],
]:
    """Train Vector-GSQ using Q, K, V/O, and MLP activation objectives."""
    seed_everything(args.seed + 1_000_000 + layer_index)
    quantizers, initial_assignments, candidate_metadata = build_block_quantizers(
        modules,
        sources,
        hessians,
        assignments,
        args,
        layer_index,
    )
    hessians.clear()
    torch.cuda.empty_cache()

    if validation_ids.shape[0] > 0:
        global_selection_ids = validation_ids
        global_selection_outputs = validation_dense_outputs
        global_selection_batch_size = args.validation_batch_size
        selection_split = "validation"
    else:
        global_selection_ids = train_ids
        global_selection_outputs = train_dense_outputs
        global_selection_batch_size = args.train_batch_size
        selection_split = "training"

    initial_weights = merge_selected_weights(
        quantizers,
        initial_assignments,
        LINEAR_NAMES,
    )
    initial_block_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        global_selection_ids,
        global_selection_outputs,
        initial_weights,
        global_selection_batch_size,
        next(iter(modules.values())).weight.device,
        True,
    )
    del initial_weights

    selected_assignments = {
        name: value.detach().clone() for name, value in initial_assignments.items()
    }
    device = next(iter(modules.values())).weight.device
    joint_scale_names = [
        name
        for name in LINEAR_NAMES
        if args.staged_joint_group_scales
        and not (
            args.staged_freeze_qk and name in {"self_attn.q_proj", "self_attn.k_proj"}
        )
    ]
    selected_group_logs = {
        name: torch.zeros(
            modules[name].out_features,
            math.ceil(modules[name].in_features / args.scale_group_size),
            device=device,
            dtype=torch.float32,
        )
        for name in joint_scale_names
    }
    schedule = make_schedule(args.schedule)
    phase_results = []
    total_started = time.time()

    for phase_index, phase in enumerate(staged_phase_specs()):
        phase_started = time.time()
        if args.staged_freeze_qk and phase.name in {"q_proj", "k_proj"}:
            phase_results.append(
                {
                    "name": phase.name,
                    "objective": phase.objective,
                    "active_linears": list(phase.active_names),
                    "frozen_linears": list(phase.frozen_names),
                    "steps": 0,
                    "steps_per_epoch": 0,
                    "epochs_equivalent": 0.0,
                    "initial_hard_mse": None,
                    "final_hard_mse": None,
                    "accepted": False,
                    "skipped": True,
                    "skip_reason": "staged_freeze_qk",
                    "training_seconds": 0.0,
                    "trace": [],
                }
            )
            emit_progress(
                "vector_gsq_phase_skipped",
                layer=layer_index,
                phase=phase.name,
                reason="staged_freeze_qk",
            )
            continue
        emit_progress(
            "vector_gsq_phase_started",
            layer=layer_index,
            phase=phase.name,
            active_linears=list(phase.active_names),
        )
        reference_weights = apply_group_scales(
            merge_selected_weights(
                quantizers,
                selected_assignments,
                phase.frozen_names,
            ),
            modules,
            {
                name: selected_group_logs[name].exp().to(torch.float16).float()
                for name in phase.frozen_names
                if name in selected_group_logs
            },
            args.scale_group_size,
        )
        train_targets = (
            torch.empty(0)
            if args.stream_dense_targets
            else collect_dense_stage_targets(
                model,
                layer_index,
                modules,
                train_ids,
                train_dense_outputs,
                phase.objective,
                args.cache_microbatch_size,
            )
        )
        validation_targets = (
            collect_dense_stage_targets(
                model,
                layer_index,
                modules,
                validation_ids,
                validation_dense_outputs,
                phase.objective,
                args.cache_microbatch_size,
            )
            if validation_ids.shape[0]
            else torch.empty(0)
        )
        if validation_ids.shape[0]:
            phase_selection_ids = validation_ids
            phase_selection_targets = validation_targets
            phase_selection_batch_size = args.validation_batch_size
        else:
            phase_selection_ids = train_ids
            phase_selection_targets = train_targets
            phase_selection_batch_size = args.train_batch_size

        active_quantizers = {name: quantizers[name] for name in phase.active_names}
        initial_active_weights = apply_group_scales(
            merge_selected_weights(
                quantizers,
                selected_assignments,
                phase.active_names,
            ),
            modules,
            {
                name: selected_group_logs[name].exp().to(torch.float16).float()
                for name in phase.active_names
                if name in selected_group_logs
            },
            args.scale_group_size,
        )
        initial_phase_weights = {**reference_weights, **initial_active_weights}
        initial_phase_loss = hidden_stage_mse(
            model,
            layer_index,
            modules,
            phase_selection_ids,
            phase_selection_targets,
            initial_phase_weights,
            phase.objective,
            phase_selection_batch_size,
        )
        del initial_active_weights, initial_phase_weights
        selected_phase_loss = initial_phase_loss
        selected_phase_assignments = {
            name: selected_assignments[name].detach().clone()
            for name in phase.active_names
        }
        active_group_logs = {
            name: torch.nn.Parameter(selected_group_logs[name].detach().clone())
            for name in phase.active_names
            if name in selected_group_logs
        }
        selected_phase_group_logs = {
            name: value.detach().clone() for name, value in active_group_logs.items()
        }
        steps = _staged_step_count(args, phase.name)
        steps_per_epoch = math.ceil(train_ids.shape[0] / args.train_batch_size)
        checkpoints = _training_checkpoints(steps, steps_per_epoch)
        optimizer = Lion(
            [quantizer.quant_logits for quantizer in active_quantizers.values()],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(steps, 1),
            eta_min=args.lr * 0.1,
        )
        scale_optimizer = (
            Lion(
                list(active_group_logs.values()),
                lr=args.staged_joint_scale_lr,
                weight_decay=0.0,
            )
            if active_group_logs
            else None
        )
        scale_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                scale_optimizer,
                T_max=max(steps, 1),
                eta_min=args.staged_joint_scale_lr * 0.1,
            )
            if scale_optimizer is not None
            else None
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            args.seed
            + 2_000_000
            + layer_index * len(staged_phase_specs())
            + phase_index
        )
        permutation = None
        trace = []

        for step in range(steps):
            progress = step / max(steps - 1, 1)
            temperature = (
                schedule.temperature_start
                + (schedule.temperature_end - schedule.temperature_start) * progress
            )
            kappa = (
                schedule.kappa_start
                + (schedule.kappa_end - schedule.kappa_start) * progress
            )
            batch_slot = step % steps_per_epoch
            if batch_slot == 0:
                permutation = torch.randperm(train_ids.shape[0], generator=generator)
            start = batch_slot * args.train_batch_size
            end = min(start + args.train_batch_size, train_ids.shape[0])
            batch_indices = permutation[start:end]
            optimizer.zero_grad(set_to_none=True)
            if scale_optimizer is not None:
                scale_optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            global_batch_size = batch_indices.numel()
            for micro_start in range(0, global_batch_size, args.train_microbatch_size):
                micro_end = min(
                    micro_start + args.train_microbatch_size,
                    global_batch_size,
                )
                micro_indices = batch_indices[micro_start:micro_end]
                hidden_batch = train_ids[micro_indices].to(device)
                soft_weights = dict(reference_weights)
                soft_active_weights = {
                    name: quantizer(temperature, kappa, add_gumbel=True)[0]
                    for name, quantizer in active_quantizers.items()
                }
                soft_active_weights = apply_group_scales(
                    soft_active_weights,
                    modules,
                    {name: value.exp() for name, value in active_group_logs.items()},
                    args.scale_group_size,
                )
                soft_weights.update(soft_active_weights)
                output = run_hidden_stage_output(
                    model,
                    layer_index,
                    modules,
                    hidden_batch,
                    soft_weights,
                    phase.objective,
                )
                dense_batch = stage_training_target_batch(
                    model,
                    layer_index,
                    modules,
                    hidden_batch,
                    train_targets,
                    micro_indices,
                    phase.objective,
                    args.stream_dense_targets,
                    output.device,
                )
                micro_loss = F.mse_loss(output.float(), dense_batch)
                fraction = micro_indices.numel() / max(global_batch_size, 1)
                (micro_loss * fraction).backward()
                train_loss += float(micro_loss.detach()) * fraction
                del (
                    hidden_batch,
                    soft_active_weights,
                    soft_weights,
                    output,
                    dense_batch,
                    micro_loss,
                )

            gradient_stats = aggregate_gradient_metrics(active_quantizers)
            scale_grad_elements = 0
            scale_grad_squares = 0.0
            scale_grad_nonfinite = 0
            for name, value in active_group_logs.items():
                if value.grad is None:
                    raise RuntimeError(f"Missing joint scale gradient for {name}")
                finite = torch.isfinite(value.grad)
                scale_grad_nonfinite += int((~finite).sum())
                scale_grad_elements += value.grad.numel()
                if bool(finite.any()):
                    scale_grad_squares += float(
                        value.grad[finite].float().square().sum()
                    )
            if scale_grad_nonfinite:
                raise FloatingPointError(
                    f"Joint group-scale gradient contains "
                    f"{scale_grad_nonfinite} non-finite values"
                )
            scale_gradient_stats = {
                "scale_grad_rms": float(
                    math.sqrt(scale_grad_squares / max(scale_grad_elements, 1))
                ),
                "scale_grad_nonfinite_rate": float(
                    scale_grad_nonfinite / max(scale_grad_elements, 1)
                ),
            }
            optimizer.step()
            scheduler.step()
            if scale_optimizer is not None:
                scale_optimizer.step()
                scale_scheduler.step()
                with torch.no_grad():
                    for name, value in active_group_logs.items():
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError(
                                f"Joint group scale became non-finite for {name}"
                            )
                        value.clamp_(-0.25, 0.25)

            if step in checkpoints:
                hard_active_weights, switch_rate, per_linear = (
                    hard_weights_and_switch_rate(
                        active_quantizers,
                        {
                            name: initial_assignments[name]
                            for name in phase.active_names
                        },
                    )
                )
                hard_active_weights = apply_group_scales(
                    hard_active_weights,
                    modules,
                    {
                        name: value.detach().exp().to(torch.float16).float()
                        for name, value in active_group_logs.items()
                    },
                    args.scale_group_size,
                )
                hard_weights = {**reference_weights, **hard_active_weights}
                hard_loss = hidden_stage_mse(
                    model,
                    layer_index,
                    modules,
                    phase_selection_ids,
                    phase_selection_targets,
                    hard_weights,
                    phase.objective,
                    phase_selection_batch_size,
                )
                if hard_loss < selected_phase_loss:
                    selected_phase_loss = hard_loss
                    selected_phase_assignments = {
                        name: quantizer.hard_weight()[1].detach().clone()
                        for name, quantizer in active_quantizers.items()
                    }
                    selected_phase_group_logs = {
                        name: value.detach().exp().to(torch.float16).float().log()
                        for name, value in active_group_logs.items()
                    }
                trace.append(
                    {
                        "step": float(step),
                        "temperature": float(temperature),
                        "kappa": float(kappa),
                        "soft_train_batch_mse": train_loss,
                        "hard_selection_mse": float(hard_loss),
                        "selected_hard_mse": float(selected_phase_loss),
                        "assignment_switch_rate": float(switch_rate),
                        "per_linear_switch_rate": per_linear,
                        "max_abs_log_group_scale": (
                            max(
                                float(value.detach().abs().max())
                                for value in active_group_logs.values()
                            )
                            if active_group_logs
                            else 0.0
                        ),
                        **aggregate_probability_metrics(
                            active_quantizers, temperature, kappa
                        ),
                        **gradient_stats,
                        **scale_gradient_stats,
                    }
                )
                del hard_active_weights, hard_weights

        selected_assignments.update(selected_phase_assignments)
        selected_group_logs.update(selected_phase_group_logs)
        final_phase_weights = {
            **reference_weights,
            **apply_group_scales(
                merge_selected_weights(
                    quantizers,
                    selected_assignments,
                    phase.active_names,
                ),
                modules,
                {
                    name: selected_group_logs[name].exp().to(torch.float16).float()
                    for name in phase.active_names
                    if name in selected_group_logs
                },
                args.scale_group_size,
            ),
        }
        final_phase_loss = hidden_stage_mse(
            model,
            layer_index,
            modules,
            phase_selection_ids,
            phase_selection_targets,
            final_phase_weights,
            phase.objective,
            phase_selection_batch_size,
        )
        phase_results.append(
            {
                "name": phase.name,
                "objective": phase.objective,
                "active_linears": list(phase.active_names),
                "frozen_linears": list(phase.frozen_names),
                "steps": int(steps),
                "steps_per_epoch": int(steps_per_epoch),
                "epochs_equivalent": float(steps / steps_per_epoch),
                "initial_hard_mse": float(initial_phase_loss),
                "final_hard_mse": float(final_phase_loss),
                "accepted": bool(selected_phase_loss < initial_phase_loss),
                "joint_group_scales": bool(active_group_logs),
                "joint_scale_learning_rate": (
                    float(args.staged_joint_scale_lr) if active_group_logs else None
                ),
                "training_seconds": time.time() - phase_started,
                "trace": trace,
            }
        )
        emit_progress(
            "vector_gsq_phase_complete",
            layer=layer_index,
            phase=phase.name,
            accepted=phase_results[-1]["accepted"],
            final_hard_mse=final_phase_loss,
        )
        for quantizer in active_quantizers.values():
            quantizer.quant_logits.grad = None
        del (
            train_targets,
            validation_targets,
            reference_weights,
            final_phase_weights,
            optimizer,
            scheduler,
            scale_optimizer,
            scale_scheduler,
            active_group_logs,
            active_quantizers,
        )
        torch.cuda.empty_cache()

    polish_stats = {
        "steps": 0,
        "accepted": False,
        "trace": [],
    }
    if args.block_polish_steps > 0:
        emit_progress(
            "vector_gsq_block_polish_started",
            layer=layer_index,
            steps=args.block_polish_steps,
        )
        polish_started = time.time()
        pre_polish_assignments = {
            name: selected_assignments[name].detach().clone() for name in LINEAR_NAMES
        }
        for name, quantizer in quantizers.items():
            anchor_quantizer_logits_(
                quantizer,
                pre_polish_assignments[name],
                max(args.std, 1e-4),
            )
        joint_group_logs = {
            name: torch.nn.Parameter(value.detach().clone())
            for name, value in selected_group_logs.items()
        }
        pre_polish_weights = apply_group_scales(
            merge_selected_weights(
                quantizers,
                pre_polish_assignments,
                LINEAR_NAMES,
            ),
            modules,
            {
                name: value.detach().exp().to(torch.float16).float()
                for name, value in joint_group_logs.items()
            },
            args.scale_group_size,
        )
        initial_polish_loss = dataset_block_mse(
            model,
            layer_index,
            modules,
            global_selection_ids,
            global_selection_outputs,
            pre_polish_weights,
            global_selection_batch_size,
            device,
            True,
        )
        del pre_polish_weights
        best_polish = snapshot_joint_hard_state(
            initial_polish_loss,
            pre_polish_assignments,
            selected_group_logs,
        )
        optimizer = Lion(
            [quantizer.quant_logits for quantizer in quantizers.values()],
            lr=args.block_polish_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.block_polish_steps, 1),
            eta_min=args.block_polish_lr * 0.1,
        )
        scale_optimizer = (
            Lion(
                list(joint_group_logs.values()),
                lr=args.staged_joint_scale_lr,
                weight_decay=0.0,
            )
            if joint_group_logs
            else None
        )
        scale_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                scale_optimizer,
                T_max=max(args.block_polish_steps, 1),
                eta_min=args.staged_joint_scale_lr * 0.1,
            )
            if scale_optimizer is not None
            else None
        )
        steps_per_epoch = math.ceil(train_ids.shape[0] / args.train_batch_size)
        checkpoints = _training_checkpoints(
            args.block_polish_steps,
            steps_per_epoch,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 2_500_000 + layer_index)
        permutation = None
        polish_trace = []
        for step in range(args.block_polish_steps):
            progress = step / max(args.block_polish_steps - 1, 1)
            temperature = (
                schedule.temperature_start
                + (schedule.temperature_end - schedule.temperature_start) * progress
            )
            kappa = (
                schedule.kappa_start
                + (schedule.kappa_end - schedule.kappa_start) * progress
            )
            batch_slot = step % steps_per_epoch
            if batch_slot == 0:
                permutation = torch.randperm(train_ids.shape[0], generator=generator)
            start = batch_slot * args.train_batch_size
            end = min(start + args.train_batch_size, train_ids.shape[0])
            batch_indices = permutation[start:end]
            optimizer.zero_grad(set_to_none=True)
            if scale_optimizer is not None:
                scale_optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for micro_start in range(
                0, batch_indices.numel(), args.train_microbatch_size
            ):
                micro_indices = batch_indices[
                    micro_start : micro_start + args.train_microbatch_size
                ]
                hidden_batch = train_ids[micro_indices].to(device)
                target = train_dense_outputs[micro_indices].to(device, torch.float32)
                loss = joint_block_refinement_loss(
                    model,
                    layer_index,
                    modules,
                    hidden_batch,
                    target,
                    quantizers,
                    joint_group_logs,
                    temperature,
                    kappa,
                    args.scale_group_size,
                )
                fraction = micro_indices.numel() / max(batch_indices.numel(), 1)
                (loss * fraction).backward()
                train_loss += float(loss.detach()) * fraction
                del hidden_batch, target, loss
            gradient_stats = aggregate_gradient_metrics(quantizers)
            scale_grad_elements = 0
            scale_grad_squares = 0.0
            scale_grad_nonfinite = 0
            for name, value in joint_group_logs.items():
                if value.grad is None:
                    raise RuntimeError(
                        f"Missing joint refinement scale gradient for {name}"
                    )
                finite = torch.isfinite(value.grad)
                scale_grad_nonfinite += int((~finite).sum())
                scale_grad_elements += value.grad.numel()
                if bool(finite.any()):
                    scale_grad_squares += float(
                        value.grad[finite].float().square().sum()
                    )
            if scale_grad_nonfinite:
                raise FloatingPointError(
                    "Joint refinement group-scale gradient contains "
                    f"{scale_grad_nonfinite} non-finite values"
                )
            optimizer.step()
            scheduler.step()
            if scale_optimizer is not None:
                scale_optimizer.step()
                scale_scheduler.step()
                with torch.no_grad():
                    for name, value in joint_group_logs.items():
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError(
                                "Joint refinement group scale became "
                                f"non-finite for {name}"
                            )
                        value.clamp_(-0.25, 0.25)
            if step in checkpoints:
                hard_weights, switch_rate, per_linear = hard_weights_and_switch_rate(
                    quantizers,
                    pre_polish_assignments,
                )
                deployable_logs = {
                    name: value.detach().exp().to(torch.float16).float().log()
                    for name, value in joint_group_logs.items()
                }
                hard_weights = apply_group_scales(
                    hard_weights,
                    modules,
                    {name: value.exp() for name, value in deployable_logs.items()},
                    args.scale_group_size,
                )
                hard_loss = dataset_block_mse(
                    model,
                    layer_index,
                    modules,
                    global_selection_ids,
                    global_selection_outputs,
                    hard_weights,
                    global_selection_batch_size,
                    device,
                    True,
                )
                current_assignments = {
                    name: quantizer.hard_weight()[1]
                    for name, quantizer in quantizers.items()
                }
                best_polish = select_better_joint_snapshot(
                    best_polish,
                    hard_loss,
                    current_assignments,
                    deployable_logs,
                )
                polish_trace.append(
                    {
                        "step": float(step),
                        "temperature": float(temperature),
                        "kappa": float(kappa),
                        "soft_train_batch_mse": train_loss,
                        "hard_selection_mse": float(hard_loss),
                        "selected_hard_mse": float(best_polish.loss),
                        "assignment_switch_rate": float(switch_rate),
                        "per_linear_switch_rate": per_linear,
                        "scale_grad_rms": float(
                            math.sqrt(scale_grad_squares / max(scale_grad_elements, 1))
                        ),
                        "scale_grad_nonfinite_rate": float(
                            scale_grad_nonfinite / max(scale_grad_elements, 1)
                        ),
                        **aggregate_probability_metrics(quantizers, temperature, kappa),
                        **gradient_stats,
                    }
                )
                del hard_weights, current_assignments, deployable_logs
        selected_assignments.update(best_polish.assignments)
        selected_group_logs.update(best_polish.log_group_scales)
        polish_stats = {
            "method": "all_linear_joint_dense_block_refinement",
            "steps": int(args.block_polish_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "epochs_equivalent": float(args.block_polish_steps / steps_per_epoch),
            "learning_rate": float(args.block_polish_lr),
            "scale_learning_rate": (
                float(args.staged_joint_scale_lr) if joint_group_logs else None
            ),
            "joint_group_scales": bool(joint_group_logs),
            "initial_hard_block_mse": float(initial_polish_loss),
            "final_hard_block_mse": float(best_polish.loss),
            "accepted": bool(best_polish.loss < initial_polish_loss),
            "training_seconds": time.time() - polish_started,
            "trace": polish_trace,
        }
        emit_progress(
            "vector_gsq_block_polish_complete",
            layer=layer_index,
            accepted=polish_stats["accepted"],
            final_hard_mse=best_polish.loss,
        )
        for quantizer in quantizers.values():
            quantizer.quant_logits.grad = None
        del (
            joint_group_logs,
            optimizer,
            scheduler,
            scale_optimizer,
            scale_scheduler,
            best_polish,
        )
        torch.cuda.empty_cache()

    final_assignments = {
        name: selected_assignments[name].detach().clone() for name in LINEAR_NAMES
    }
    final_group_scales = {
        name: value.exp().to(torch.float16).to(torch.float32)
        for name, value in selected_group_logs.items()
    }
    final_weights = apply_group_scales(
        merge_selected_weights(
            quantizers,
            final_assignments,
            LINEAR_NAMES,
        ),
        modules,
        final_group_scales,
        args.scale_group_size,
    )
    float_candidate_block_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        global_selection_ids,
        global_selection_outputs,
        final_weights,
        global_selection_batch_size,
        device,
        True,
    )
    del final_weights, quantizers
    torch.cuda.empty_cache()

    baseline_deployable_weights = reconstruct_logical_block_weights(
        sources,
        initial_assignments,
        modules,
        device,
        {},
        args.scale_group_size,
    )
    serialized_initial_block_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        global_selection_ids,
        global_selection_outputs,
        baseline_deployable_weights,
        global_selection_batch_size,
        device,
        True,
    )
    candidate_deployable_weights = reconstruct_logical_block_weights(
        sources,
        final_assignments,
        modules,
        device,
        final_group_scales,
        args.scale_group_size,
    )
    serialized_candidate_block_loss = dataset_block_mse(
        model,
        layer_index,
        modules,
        global_selection_ids,
        global_selection_outputs,
        candidate_deployable_weights,
        global_selection_batch_size,
        device,
        True,
    )
    global_gate_passed = bool(
        math.isfinite(serialized_candidate_block_loss)
        and serialized_candidate_block_loss < serialized_initial_block_loss
    )
    if global_gate_passed:
        final_weights = candidate_deployable_weights
        del baseline_deployable_weights
    else:
        final_weights = baseline_deployable_weights
        del candidate_deployable_weights
    if not global_gate_passed:
        final_assignments = {
            name: value.detach().clone() for name, value in initial_assignments.items()
        }
        final_group_scales = {}
    final_block_loss = (
        serialized_candidate_block_loss
        if global_gate_passed
        else serialized_initial_block_loss
    )
    final_switch_rate = sum(
        float((final_assignments[name] != initial_assignments[name]).sum())
        for name in LINEAR_NAMES
    ) / max(sum(value.numel() for value in initial_assignments.values()), 1)
    scale_count = sum(value.numel() for value in final_group_scales.values())
    weight_count = sum(value.numel() for value in final_weights.values())
    return (
        final_weights,
        final_assignments,
        final_group_scales,
        {
            "objective": "staged",
            "teacher_reference": "original_dense_network",
            "schedule": asdict(schedule),
            "candidate_metadata": candidate_metadata,
            "selection_split": selection_split,
            "train_sequences": int(train_ids.shape[0]),
            "validation_sequences": int(validation_ids.shape[0]),
            "initial_hard_block_mse": float(initial_block_loss),
            "candidate_hard_block_mse": float(float_candidate_block_loss),
            "serialized_initial_hard_block_mse": float(serialized_initial_block_loss),
            "serialized_candidate_hard_block_mse": float(
                serialized_candidate_block_loss
            ),
            "logical_checkpoint_gate_passed": bool(global_gate_passed),
            "final_hard_block_mse": float(final_block_loss),
            "hard_block_mse_relative_change": float(
                (final_block_loss - serialized_initial_block_loss)
                / max(serialized_initial_block_loss, 1e-30)
            ),
            "accepted_training": bool(global_gate_passed),
            "global_validation_gate_passed": bool(global_gate_passed),
            "final_assignment_switch_rate": float(final_switch_rate),
            "training_seconds": time.time() - total_started,
            "joint_group_scales": {
                "enabled": bool(args.staged_joint_group_scales),
                "active_linears": list(final_group_scales),
                "group_size": int(args.scale_group_size),
                "learning_rate": float(args.staged_joint_scale_lr),
                "scale_count": int(scale_count),
                "metadata_bits": int(scale_count * 16),
                "metadata_bpp_overhead": float(scale_count * 16 / max(weight_count, 1)),
                "full_logical_checkpoint_gate": True,
            },
            "phases": phase_results,
            "block_polish": polish_stats,
        },
    )


@torch.inference_mode()
def _row_scale_block_mse(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    dense_outputs: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    row_scales: Dict[str, torch.Tensor],
    batch_size: int,
) -> float:
    squared_error = 0.0
    elements = 0
    device = next(iter(modules.values())).weight.device
    for start in range(0, hidden_states.shape[0], batch_size):
        end = min(start + batch_size, hidden_states.shape[0])
        output = run_hidden_block_with_row_scales(
            model,
            layer_index,
            modules,
            hidden_states[start:end].to(device),
            weights,
            row_scales,
        ).float()
        target = dense_outputs[start:end].to(output.device, torch.float32)
        squared_error += float((output - target).square().sum())
        elements += output.numel()
    return squared_error / max(elements, 1)


def finetune_fixed_assignment_row_scales(
    model,
    layer_index: int,
    modules,
    train_ids: torch.Tensor,
    train_dense_outputs: torch.Tensor,
    validation_ids: torch.Tensor,
    validation_dense_outputs: torch.Tensor,
    sources: Dict[str, torch.nn.Module],
    weights: Dict[str, torch.Tensor],
    assignments: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Refine existing row-normalization factors without changing codes."""
    if validation_ids.shape[0] > 0:
        selection_ids = validation_ids
        selection_outputs = validation_dense_outputs
        selection_batch_size = args.validation_batch_size
        selection_split = "validation"
    else:
        selection_ids = train_ids
        selection_outputs = train_dense_outputs
        selection_batch_size = args.train_batch_size
        selection_split = "training"
    device = next(iter(modules.values())).weight.device
    active_names = [
        name
        for name in LINEAR_NAMES
        if not (
            args.staged_freeze_qk and name in {"self_attn.q_proj", "self_attn.k_proj"}
        )
    ]
    log_scales = {
        name: torch.nn.Parameter(
            torch.zeros(module.out_features, device=device, dtype=torch.float32)
        )
        for name, module in modules.items()
        if name in active_names
    }
    unit_scales = {name: value.detach().exp() for name, value in log_scales.items()}
    initial_loss = _row_scale_block_mse(
        model,
        layer_index,
        modules,
        selection_ids,
        selection_outputs,
        weights,
        unit_scales,
        selection_batch_size,
    )
    selected_loss = initial_loss
    selected_logs = {name: value.detach().clone() for name, value in log_scales.items()}
    optimizer = Lion(
        list(log_scales.values()),
        lr=args.scale_finetune_lr,
        weight_decay=0.0,
    )
    steps_per_epoch = math.ceil(train_ids.shape[0] / args.train_batch_size)
    checkpoints = _training_checkpoints(
        args.scale_finetune_steps,
        steps_per_epoch,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 3_000_000 + layer_index)
    permutation = None
    trace = []
    started = time.time()
    for step in range(args.scale_finetune_steps):
        batch_slot = step % steps_per_epoch
        if batch_slot == 0:
            permutation = torch.randperm(train_ids.shape[0], generator=generator)
        start = batch_slot * args.train_batch_size
        end = min(start + args.train_batch_size, train_ids.shape[0])
        batch_indices = permutation[start:end]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for micro_start in range(0, batch_indices.numel(), args.train_microbatch_size):
            micro_indices = batch_indices[
                micro_start : micro_start + args.train_microbatch_size
            ]
            scales = {name: value.exp() for name, value in log_scales.items()}
            output = run_hidden_block_with_row_scales(
                model,
                layer_index,
                modules,
                train_ids[micro_indices].to(device),
                weights,
                scales,
            )
            target = train_dense_outputs[micro_indices].to(output.device, torch.float32)
            loss = F.mse_loss(output.float(), target)
            fraction = micro_indices.numel() / max(batch_indices.numel(), 1)
            (loss * fraction).backward()
            batch_loss += float(loss.detach()) * fraction
        optimizer.step()
        with torch.no_grad():
            for value in log_scales.values():
                value.clamp_(-0.25, 0.25)
        if step in checkpoints:
            hard_scales = {
                name: value.detach().exp() for name, value in log_scales.items()
            }
            hard_loss = _row_scale_block_mse(
                model,
                layer_index,
                modules,
                selection_ids,
                selection_outputs,
                weights,
                hard_scales,
                selection_batch_size,
            )
            if hard_loss < selected_loss:
                selected_loss = hard_loss
                selected_logs = {
                    name: value.detach().clone() for name, value in log_scales.items()
                }
            trace.append(
                {
                    "step": float(step),
                    "soft_train_batch_mse": batch_loss,
                    "hard_selection_mse": float(hard_loss),
                    "selected_hard_mse": float(selected_loss),
                    "max_abs_log_scale": max(
                        float(value.detach().abs().max())
                        for value in log_scales.values()
                    ),
                }
            )
    float_gate_passed = selected_loss < initial_loss
    selected_scales = {name: value.exp() for name, value in selected_logs.items()}
    serialized_initial_loss = None
    serialized_candidate_loss = None
    serialization_gate_passed = False
    if float_gate_passed:
        baseline_states = {
            name: logical_state(
                sources[name],
                assignments[name],
                tuple(modules[name].weight.shape),
            )[0]
            for name in LINEAR_NAMES
        }
        baseline_weights = {
            name: reconstruct_logical_state(state, device)
            for name, state in baseline_states.items()
        }
        serialized_initial_loss = dataset_block_mse(
            model,
            layer_index,
            modules,
            selection_ids,
            selection_outputs,
            baseline_weights,
            selection_batch_size,
            device,
            True,
        )
        del baseline_states, baseline_weights
        torch.cuda.empty_cache()

        original_row_norms = {
            name: source.normalizer.norms[1].detach().clone()
            for name, source in sources.items()
        }
        with torch.no_grad():
            for name, scale in selected_scales.items():
                row_norm = sources[name].normalizer.norms[1]
                row_norm.mul_(scale.to(row_norm).reshape_as(row_norm))
        candidate_states = {
            name: logical_state(
                sources[name],
                assignments[name],
                tuple(modules[name].weight.shape),
            )[0]
            for name in LINEAR_NAMES
        }
        candidate_weights = {
            name: reconstruct_logical_state(state, device)
            for name, state in candidate_states.items()
        }
        serialized_candidate_loss = dataset_block_mse(
            model,
            layer_index,
            modules,
            selection_ids,
            selection_outputs,
            candidate_weights,
            selection_batch_size,
            device,
            True,
        )
        serialization_gate_passed = serialized_candidate_loss < serialized_initial_loss
        del candidate_states, candidate_weights
        torch.cuda.empty_cache()
        if serialization_gate_passed:
            with torch.no_grad():
                for name, scale in selected_scales.items():
                    weights[name].mul_(scale.to(weights[name])[:, None])
        else:
            with torch.no_grad():
                for name, original in original_row_norms.items():
                    sources[name].normalizer.norms[1].copy_(original)
    accepted = float_gate_passed and serialization_gate_passed
    return weights, {
        "scope": "block_output",
        "fixed_assignments": True,
        "steps": int(args.scale_finetune_steps),
        "learning_rate": float(args.scale_finetune_lr),
        "selection_split": selection_split,
        "initial_hard_block_mse": float(initial_loss),
        "float_candidate_hard_block_mse": float(selected_loss),
        "float_gate_passed": bool(float_gate_passed),
        "serialized_initial_hard_block_mse": (
            float(serialized_initial_loss)
            if serialized_initial_loss is not None
            else None
        ),
        "serialized_candidate_hard_block_mse": (
            float(serialized_candidate_loss)
            if serialized_candidate_loss is not None
            else None
        ),
        "serialization_gate_passed": bool(serialization_gate_passed),
        "final_hard_block_mse": float(
            serialized_candidate_loss
            if accepted
            else (
                serialized_initial_loss
                if serialized_initial_loss is not None
                else initial_loss
            )
        ),
        "accepted": bool(accepted),
        "training_seconds": time.time() - started,
        "trace": trace,
    }


@torch.inference_mode()
def _group_scale_block_mse(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    dense_outputs: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    group_scales: Dict[str, torch.Tensor],
    group_size: int,
    batch_size: int,
) -> float:
    squared_error = 0.0
    elements = 0
    device = next(iter(modules.values())).weight.device
    for start in range(0, hidden_states.shape[0], batch_size):
        end = min(start + batch_size, hidden_states.shape[0])
        output = run_hidden_block_with_group_scales(
            model,
            layer_index,
            modules,
            hidden_states[start:end].to(device),
            weights,
            group_scales,
            group_size,
        ).float()
        target = dense_outputs[start:end].to(output.device, torch.float32)
        squared_error += float((output - target).square().sum())
        elements += output.numel()
    return squared_error / max(elements, 1)


def finetune_fixed_assignment_group_scales(
    model,
    layer_index: int,
    modules,
    train_ids: torch.Tensor,
    train_dense_outputs: torch.Tensor,
    validation_ids: torch.Tensor,
    validation_dense_outputs: torch.Tensor,
    sources: Dict[str, torch.nn.Module],
    weights: Dict[str, torch.Tensor],
    assignments: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, object],
]:
    """Train deployable per-row/input-group scales with codes held fixed."""
    if validation_ids.shape[0] > 0:
        selection_ids = validation_ids
        selection_outputs = validation_dense_outputs
        selection_batch_size = args.validation_batch_size
        selection_split = "validation"
    else:
        selection_ids = train_ids
        selection_outputs = train_dense_outputs
        selection_batch_size = args.train_batch_size
        selection_split = "training"
    group_size = int(args.scale_group_size)
    device = next(iter(modules.values())).weight.device
    active_names = [
        name
        for name in LINEAR_NAMES
        if not (
            args.staged_freeze_qk and name in {"self_attn.q_proj", "self_attn.k_proj"}
        )
    ]
    log_scales = {
        name: torch.nn.Parameter(
            torch.zeros(
                module.out_features,
                math.ceil(module.in_features / group_size),
                device=device,
                dtype=torch.float32,
            )
        )
        for name, module in modules.items()
        if name in active_names
    }
    unit_scales = {name: value.detach().exp() for name, value in log_scales.items()}
    initial_loss = _group_scale_block_mse(
        model,
        layer_index,
        modules,
        selection_ids,
        selection_outputs,
        weights,
        unit_scales,
        group_size,
        selection_batch_size,
    )
    selected_loss = initial_loss
    selected_logs = {name: value.detach().clone() for name, value in log_scales.items()}
    optimizer = Lion(
        list(log_scales.values()),
        lr=args.scale_finetune_lr,
        weight_decay=0.0,
    )
    steps_per_epoch = math.ceil(train_ids.shape[0] / args.train_batch_size)
    checkpoints = _training_checkpoints(args.scale_finetune_steps, steps_per_epoch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 4_000_000 + layer_index)
    permutation = None
    trace = []
    started = time.time()
    for step in range(args.scale_finetune_steps):
        batch_slot = step % steps_per_epoch
        if batch_slot == 0:
            permutation = torch.randperm(train_ids.shape[0], generator=generator)
        start = batch_slot * args.train_batch_size
        end = min(start + args.train_batch_size, train_ids.shape[0])
        batch_indices = permutation[start:end]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for micro_start in range(
            0,
            batch_indices.numel(),
            args.train_microbatch_size,
        ):
            micro_indices = batch_indices[
                micro_start : micro_start + args.train_microbatch_size
            ]
            scales = {name: value.exp() for name, value in log_scales.items()}
            output = run_hidden_block_with_group_scales(
                model,
                layer_index,
                modules,
                train_ids[micro_indices].to(device),
                weights,
                scales,
                group_size,
            )
            target = train_dense_outputs[micro_indices].to(
                output.device,
                torch.float32,
            )
            loss = F.mse_loss(output.float(), target)
            fraction = micro_indices.numel() / max(batch_indices.numel(), 1)
            (loss * fraction).backward()
            batch_loss += float(loss.detach()) * fraction
        optimizer.step()
        with torch.no_grad():
            for value in log_scales.values():
                value.clamp_(-0.25, 0.25)
        if step in checkpoints:
            hard_scales = {
                name: value.detach().exp() for name, value in log_scales.items()
            }
            hard_loss = _group_scale_block_mse(
                model,
                layer_index,
                modules,
                selection_ids,
                selection_outputs,
                weights,
                hard_scales,
                group_size,
                selection_batch_size,
            )
            if hard_loss < selected_loss:
                selected_loss = hard_loss
                selected_logs = {
                    name: value.detach().clone() for name, value in log_scales.items()
                }
            trace.append(
                {
                    "step": float(step),
                    "soft_train_batch_mse": batch_loss,
                    "hard_selection_mse": float(hard_loss),
                    "selected_hard_mse": float(selected_loss),
                    "max_abs_log_scale": max(
                        float(value.detach().abs().max())
                        for value in log_scales.values()
                    ),
                }
            )

    float_gate_passed = selected_loss < initial_loss
    selected_scales = {name: value.exp() for name, value in selected_logs.items()}
    serialized_initial_loss = None
    serialized_candidate_loss = None
    serialization_gate_passed = False
    deployed_scales: Dict[str, torch.Tensor] = {}
    selected_weights = weights
    if float_gate_passed:
        baseline_states = {
            name: logical_state(
                sources[name],
                assignments[name],
                tuple(modules[name].weight.shape),
            )[0]
            for name in LINEAR_NAMES
        }
        baseline_weights = {
            name: reconstruct_logical_state(state, device)
            for name, state in baseline_states.items()
        }
        serialized_initial_loss = dataset_block_mse(
            model,
            layer_index,
            modules,
            selection_ids,
            selection_outputs,
            baseline_weights,
            selection_batch_size,
            device,
            True,
        )
        candidate_states = {
            name: logical_state(
                sources[name],
                assignments[name],
                tuple(modules[name].weight.shape),
                group_scale=selected_scales.get(name),
                group_size=(group_size if name in selected_scales else None),
            )[0]
            for name in LINEAR_NAMES
        }
        candidate_weights = {
            name: reconstruct_logical_state(state, device)
            for name, state in candidate_states.items()
        }
        serialized_candidate_loss = dataset_block_mse(
            model,
            layer_index,
            modules,
            selection_ids,
            selection_outputs,
            candidate_weights,
            selection_batch_size,
            device,
            True,
        )
        serialization_gate_passed = serialized_candidate_loss < serialized_initial_loss
        if serialization_gate_passed:
            selected_weights = candidate_weights
            deployed_scales = {
                name: state["group_scale"].to(device=device, dtype=torch.float32)
                for name, state in candidate_states.items()
                if "group_scale" in state
            }
        del baseline_states, baseline_weights, candidate_states
        if not serialization_gate_passed:
            del candidate_weights
        torch.cuda.empty_cache()
    accepted = float_gate_passed and serialization_gate_passed
    scale_count = sum(value.numel() for value in log_scales.values())
    weight_count = sum(value.numel() for value in weights.values())
    return (
        selected_weights,
        deployed_scales,
        {
            "scope": "block_output",
            "mode": "group",
            "fixed_assignments": True,
            "active_linears": active_names,
            "group_size": group_size,
            "scale_count": int(scale_count),
            "metadata_bits": int(scale_count * 16),
            "metadata_bpp_overhead": float(scale_count * 16 / max(weight_count, 1)),
            "steps": int(args.scale_finetune_steps),
            "learning_rate": float(args.scale_finetune_lr),
            "selection_split": selection_split,
            "initial_hard_block_mse": float(initial_loss),
            "float_candidate_hard_block_mse": float(selected_loss),
            "float_gate_passed": bool(float_gate_passed),
            "serialized_initial_hard_block_mse": (
                float(serialized_initial_loss)
                if serialized_initial_loss is not None
                else None
            ),
            "serialized_candidate_hard_block_mse": (
                float(serialized_candidate_loss)
                if serialized_candidate_loss is not None
                else None
            ),
            "serialization_gate_passed": bool(serialization_gate_passed),
            "final_hard_block_mse": float(
                serialized_candidate_loss
                if accepted
                else (
                    serialized_initial_loss
                    if serialized_initial_loss is not None
                    else initial_loss
                )
            ),
            "accepted": bool(accepted),
            "training_seconds": time.time() - started,
            "trace": trace,
        },
    )


def logical_state(
    source,
    assignments: torch.Tensor,
    shape: Tuple[int, int],
    group_scale: torch.Tensor | None = None,
    group_size: int | None = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    row_norm = source.normalizer.norms[1].detach().to("cpu", torch.float16)
    col_norm = source.normalizer.norms[0].detach().to("cpu", torch.float16)
    codebook = source.codebook.detach().to("cpu", torch.float16)
    # PyTorch 2.3 cannot serialize uint16 storages. Physical checkpoint dtype
    # is therefore int32; logical bpp below still uses the true assignment_bits.
    assignment_cpu = assignments.detach().to("cpu", torch.int32)
    assignment_bits = math.ceil(math.log2(codebook.shape[0]))
    bits = assignment_cpu.numel() * assignment_bits
    bits += codebook.numel() * 16
    bits += (row_norm.numel() + col_norm.numel()) * 16
    state = {
        "assignments": assignment_cpu,
        "codebook": codebook,
        "row_norm": row_norm,
        "col_norm": col_norm,
        "shape": torch.tensor(shape, dtype=torch.int32),
        "vector_dim": torch.tensor(codebook.shape[1], dtype=torch.int32),
        "assignment_bits": torch.tensor(assignment_bits, dtype=torch.int32),
    }
    if group_scale is not None:
        if group_size is None or group_size <= 0:
            raise ValueError("positive group_size is required with group_scale")
        expected_shape = (shape[0], math.ceil(shape[1] / group_size))
        if tuple(group_scale.shape) != expected_shape:
            raise ValueError(
                f"group_scale shape {tuple(group_scale.shape)} does not match "
                f"expected {expected_shape}"
            )
        stored_group_scale = group_scale.detach().to("cpu", torch.float16)
        state["group_scale"] = stored_group_scale
        state["group_size"] = torch.tensor(group_size, dtype=torch.int32)
        bits += stored_group_scale.numel() * 16
    return state, bits

@dataclass(frozen=True)
class ResumeSummary:
    completed_layers: int
    total_logical_bits: int
    total_quantized_weights: int


_RESUME_COMPATIBILITY_KEYS = (
    "model",
    "stage",
    "seed",
    "max_layers",
    "calib_sequences",
    "calib_sequence_length",
    "gptq_token_offset",
    "train_sequences",
    "train_sequence_length",
    "train_segment_length",
    "train_batch_size",
    "train_microbatch_size",
    "validation_sequences",
    "validation_token_offset",
    "validation_segment_length",
    "vector_dim",
    "bits_per_weight",
    "lloyd_iters",
    "local_candidates",
    "gptq_percdamp",
    "gptq_candidates",
    "steps",
    "gsq_objective",
    "stage_steps_qk",
    "stage_steps_vo",
    "stage_steps_mlp",
    "staged_freeze_qk",
    "staged_joint_group_scales",
    "staged_joint_scale_lr",
    "scale_finetune_steps",
    "scale_finetune_lr",
    "scale_finetune_mode",
    "scale_group_size",
    "block_polish_steps",
    "block_polish_lr",
    "early_layers",
    "early_local_candidates",
    "early_steps",
    "lr",
    "weight_decay",
    "std",
    "strength",
    "schedule",
    "half_kmeans",
)


def logical_state_usage(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Return logical bits and represented scalar weights for a saved state."""
    required = {
        "assignments",
        "codebook",
        "row_norm",
        "col_norm",
        "shape",
        "vector_dim",
        "assignment_bits",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"logical state is missing keys: {sorted(missing)}")
    shape = tuple(int(value) for value in state["shape"])
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"invalid logical weight shape: {shape}")
    if "group_scale" in state:
        if "group_size" not in state:
            raise ValueError("group_scale checkpoint is missing group_size")
    return logical_state_bits(state), int(math.prod(shape))


def validate_resume_artifacts(
    checkpoint: Dict[str, object],
    result: Dict[str, object],
    args: argparse.Namespace,
    model_family: str,
) -> ResumeSummary:
    """Validate that JSON and logical checkpoint describe one safe prefix."""
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("layers"), dict
    ):
        raise ValueError("resume checkpoint does not contain a layers mapping")
    checkpoint_layers = checkpoint["layers"]
    try:
        layer_indices = sorted(int(key) for key in checkpoint_layers)
    except (TypeError, ValueError) as error:
        raise ValueError("resume checkpoint layer keys must be integers") from error
    expected_indices = list(range(len(layer_indices)))
    if layer_indices != expected_indices:
        raise ValueError(
            "resume checkpoint layers must form a contiguous prefix from zero"
        )
    if len(layer_indices) > args.max_layers:
        raise ValueError("resume checkpoint contains more layers than --max-layers")

    result_layers = result.get("layers")
    if not isinstance(result_layers, list):
        raise ValueError("resume result JSON does not contain a layers list")
    result_indices = [entry.get("layer") for entry in result_layers]
    if result_indices != expected_indices:
        raise ValueError(
            "resume checkpoint and result JSON disagree on completed layers"
        )
    if result.get("model_family") != model_family:
        raise ValueError("resume model_family does not match the loaded model")

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint does not contain metadata")
    if metadata.get("model_family") != model_family:
        raise ValueError(
            "resume checkpoint model_family does not match the loaded model"
        )
    for key in _RESUME_COMPATIBILITY_KEYS:
        if key not in metadata or not hasattr(args, key):
            continue
        stored = metadata[key]
        current = getattr(args, key)
        if stored != current:
            raise ValueError(
                f"resume configuration mismatch for {key}: "
                f"checkpoint={stored!r}, current={current!r}"
            )

    expected_linears = set(LINEAR_NAMES)
    total_bits = 0
    total_weights = 0
    for layer_index in expected_indices:
        layer = checkpoint_layers[str(layer_index)]
        if not isinstance(layer, dict) or set(layer) != expected_linears:
            raise ValueError(f"resume layer {layer_index} has an invalid Linear set")
        for state in layer.values():
            logical_bits, scalar_weights = logical_state_usage(state)
            total_bits += logical_bits
            total_weights += scalar_weights
    return ResumeSummary(
        completed_layers=len(layer_indices),
        total_logical_bits=total_bits,
        total_quantized_weights=total_weights,
    )


@torch.inference_mode()
def restore_logical_layer_weights(
    modules: Dict[str, torch.nn.Module],
    checkpoint_layer: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> None:
    """Restore one completed block exactly from its deployable logical state."""
    expected_linears = set(LINEAR_NAMES)
    if set(checkpoint_layer) != expected_linears:
        raise ValueError("resume checkpoint layer has an invalid Linear set")
    if not expected_linears.issubset(modules):
        raise ValueError("loaded model block has an invalid Linear set")
    for name in LINEAR_NAMES:
        module = modules[name]
        state = checkpoint_layer[name]
        stored_shape = tuple(int(value) for value in state["shape"])
        if stored_shape != tuple(module.weight.shape):
            raise ValueError(
                f"resume shape mismatch for {name}: "
                f"checkpoint={stored_shape}, model={tuple(module.weight.shape)}"
            )
        weight = reconstruct_logical_state(state, device)
        module.weight.copy_(weight.to(module.weight.device, module.weight.dtype))


@torch.inference_mode()
def replay_completed_prefix(
    model,
    checkpoint: Dict[str, object],
    completed_layers: int,
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    gptq_inputs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Restore completed weights and rebuild every cached prefix activation."""
    for layer_index in range(completed_layers):
        move_layer_for_quantization(
            model,
            layer_index,
            device,
            enabled=args.layerwise_cpu_offload,
        )
        modules = get_linear_modules(model, layer_index)
        restore_logical_layer_weights(
            modules,
            checkpoint["layers"][str(layer_index)],
            device,
        )
        if train_inputs.shape[0]:
            propagate_hidden_cache_in_place(
                model,
                layer_index,
                modules,
                train_inputs,
                args.cache_microbatch_size,
                device,
            )
        if validation_inputs.shape[0]:
            propagate_hidden_cache_in_place(
                model,
                layer_index,
                modules,
                validation_inputs,
                args.cache_microbatch_size,
                device,
            )
        propagate_hidden_cache_in_place(
            model,
            layer_index,
            modules,
            gptq_inputs,
            args.cache_microbatch_size,
            device,
        )
        offload_layer_after_quantization(
            model,
            layer_index,
            enabled=args.layerwise_cpu_offload,
        )
        emit_progress(
            "resume_layer_replayed",
            layer=layer_index,
            completed_layers=completed_layers,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.inference_mode()
def reconstruct_logical_block_weights(
    sources: Dict[str, torch.nn.Module],
    assignments: Dict[str, torch.Tensor],
    modules: Dict[str, torch.nn.Module],
    device: torch.device,
    group_scales: Dict[str, torch.Tensor],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """Reconstruct the exact FP16-metadata weights used at deployment."""
    weights = {}
    for name in LINEAR_NAMES:
        state, _ = logical_state(
            sources[name],
            assignments[name],
            tuple(modules[name].weight.shape),
            group_scale=group_scales.get(name),
            group_size=(group_size if name in group_scales else None),
        )
        weights[name] = reconstruct_logical_state(state, device)
    return weights


@torch.inference_mode()
def evaluate_wikitext2(model, tokenizer, seqlen: int, batch_size: int) -> float:
    dataset = get_dataset("wikitext2", tokenizer)
    loader = prepare_test_dataloader(
        dataset["test"],
        tokenizer,
        seqlen=seqlen,
        batch_size=batch_size,
    )
    return evaluate_ppl(model, model.config.pad_token_id, loader)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if any(
        value < 0
        for value in (
            args.stage_steps_qk,
            args.stage_steps_vo,
            args.stage_steps_mlp,
            args.scale_finetune_steps,
            args.block_polish_steps,
        )
    ):
        raise ValueError("Staged and scale fine-tuning steps must be non-negative")
    if args.scale_finetune_lr <= 0:
        raise ValueError("scale-finetune-lr must be positive")
    if args.staged_joint_scale_lr <= 0:
        raise ValueError("staged-joint-scale-lr must be positive")
    if args.scale_group_size <= 0:
        raise ValueError("scale-group-size must be positive")
    if args.block_polish_lr <= 0:
        raise ValueError("block-polish-lr must be positive")
    if args.train_batch_size <= 0 or args.train_microbatch_size <= 0:
        raise ValueError("Training batch sizes must be positive")
    if args.train_microbatch_size > args.train_batch_size:
        raise ValueError("train-microbatch-size cannot exceed train-batch-size")
    if args.cache_microbatch_size <= 0 or args.validation_batch_size <= 0:
        raise ValueError("Cache and validation batch sizes must be positive")
    if not 1 <= args.gptq_concurrency <= len(LINEAR_NAMES):
        raise ValueError(f"gptq-concurrency must be in [1, {len(LINEAR_NAMES)}]")
    if args.gpu_reservation_mib < 0:
        raise ValueError("gpu-reservation-mib must be non-negative")
    if args.early_layers < 0 or args.early_layers > args.max_layers:
        raise ValueError("early-layers must be in [0, max-layers]")
    if args.early_layers > 0 and (
        args.early_local_candidates <= 0 or args.early_steps <= 0
    ):
        raise ValueError(
            "early-local-candidates and early-steps must be positive when "
            "early-layers is enabled"
        )
    if args.gsq_objective == "staged" and args.stage != "gsq":
        raise ValueError("gsq-objective=staged requires stage=gsq")
    if args.gsq_objective == "staged" and not args.use_hidden_cache:
        raise ValueError("gsq-objective=staged requires use-hidden-cache")
    if args.resume and not args.use_hidden_cache:
        raise ValueError("resume currently requires use-hidden-cache")
    if args.stream_dense_targets:
        if args.stage != "gsq" or args.gsq_objective != "staged":
            raise ValueError(
                "stream-dense-targets requires stage=gsq and gsq-objective=staged"
            )
        if not args.use_hidden_cache:
            raise ValueError("stream-dense-targets requires use-hidden-cache")
        if args.validation_sequences <= 0:
            raise ValueError(
                "stream-dense-targets requires a non-empty validation split"
            )
        if args.scale_finetune_steps > 0 or args.block_polish_steps > 0:
            raise ValueError(
                "stream-dense-targets currently requires scale-finetune-steps=0 "
                "and block-polish-steps=0"
            )
    if args.staged_freeze_qk and args.gsq_objective != "staged":
        raise ValueError("staged-freeze-qk requires gsq-objective=staged")
    validate_block_refinement_contract(
        args.staged_freeze_qk,
        args.block_polish_steps,
    )
    if args.staged_joint_group_scales and args.gsq_objective != "staged":
        raise ValueError("staged-joint-group-scales requires gsq-objective=staged")
    if args.staged_joint_group_scales and args.scale_finetune_mode != "group":
        raise ValueError("staged-joint-group-scales requires scale-finetune-mode=group")
    if args.staged_joint_group_scales and args.scale_finetune_steps > 0:
        raise ValueError(
            "joint staged scales already consume the group-scale metadata; "
            "set scale-finetune-steps=0"
        )
    if args.scale_finetune_steps > 0 and not args.use_hidden_cache:
        raise ValueError("scale fine-tuning requires use-hidden-cache")
    if args.scale_finetune_steps > 0 and args.stage != "gsq":
        raise ValueError("scale fine-tuning requires stage=gsq")
    if args.layerwise_cpu_offload:
        if args.hf_device_map:
            raise ValueError(
                "layerwise-cpu-offload and hf-device-map are mutually exclusive"
            )
        if not args.use_hidden_cache:
            raise ValueError("layerwise-cpu-offload requires use-hidden-cache")
        if args.stage != "gsq":
            raise ValueError("layerwise-cpu-offload currently requires stage=gsq")
        if torch.device(args.device).type != "cuda":
            raise ValueError("layerwise-cpu-offload requires a CUDA device")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gpu_reservation = None
    if args.gpu_reservation_mib:
        if device.type != "cuda":
            raise ValueError("gpu-reservation-mib requires a CUDA device")
        gpu_reservation = torch.empty(
            args.gpu_reservation_mib * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        emit_progress(
            "gpu_reserved",
            device=str(device),
            reservation_mib=args.gpu_reservation_mib,
        )
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        missing_resume_files = [
            str(path) for path in (output_path, checkpoint_path) if not path.is_file()
        ]
        if missing_resume_files:
            raise FileNotFoundError(
                "resume requires existing result and checkpoint files: "
                f"{missing_resume_files}"
            )
    depths = {int(value) for value in args.ppl_depths.split(",") if value.strip()}
    if args.layerwise_cpu_offload and depths:
        raise ValueError(
            "layerwise-cpu-offload does not support inline prefix PPL; "
            "set ppl-depths to an empty string"
        )

    payload = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    validate_token_cache_identity(payload.get("identity"), args.model)
    token_ids = payload["input_ids"]
    train_source_ids = token_ids[
        : args.train_sequences, : args.train_sequence_length
    ].contiguous()
    validation_source_ids = token_ids[
        args.validation_token_offset : args.validation_token_offset
        + args.validation_sequences,
        : args.train_sequence_length,
    ].contiguous()
    gptq_ids = token_ids[
        args.gptq_token_offset : args.gptq_token_offset + args.calib_sequences,
        : args.calib_sequence_length,
    ].contiguous()
    if train_source_ids.shape[0] != args.train_sequences:
        raise ValueError("Token cache does not contain the requested training split")
    if validation_source_ids.shape[0] != args.validation_sequences:
        raise ValueError("Token cache does not contain the requested validation split")
    if gptq_ids.shape[0] != args.calib_sequences:
        raise ValueError("Token cache does not contain the requested GPTQ split")
    train_ids = split_into_independent_segments(
        train_source_ids,
        args.train_segment_length,
        "training",
    )
    validation_ids = split_into_independent_segments(
        validation_source_ids,
        args.validation_segment_length,
        "validation",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    emit_progress(
        "model_load_started",
        model=args.model,
        layerwise_cpu_offload=args.layerwise_cpu_offload,
    )
    model_load_kwargs = build_model_load_kwargs(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_kwargs,
    )
    model_family = infer_model_family(model.config)
    if not args.hf_device_map and not args.layerwise_cpu_offload:
        model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    emit_progress(
        "model_load_complete",
        model=args.model,
        model_family=model_family,
        layers=len(model.model.layers),
        layerwise_cpu_offload=args.layerwise_cpu_offload,
    )
    if args.use_hidden_cache:
        emit_progress("hidden_cache_build_started")
        train_inputs = (
            embed_token_split(model, train_ids, args.cache_microbatch_size, device)
            if args.stage == "gsq"
            else torch.empty(0)
        )
        validation_inputs = (
            embed_token_split(model, validation_ids, args.cache_microbatch_size, device)
            if validation_ids.shape[0]
            else torch.empty(0)
        )
        gptq_inputs = embed_token_split(
            model, gptq_ids, args.cache_microbatch_size, device
        )
        emit_progress(
            "hidden_cache_build_complete",
            train_shape=list(train_inputs.shape),
            validation_shape=list(validation_inputs.shape),
            gptq_shape=list(gptq_inputs.shape),
        )
    else:
        train_inputs = train_ids
        validation_inputs = validation_ids
        gptq_inputs = gptq_ids

    if args.resume:
        with output_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        resume_summary = validate_resume_artifacts(
            checkpoint,
            result,
            args,
            model_family,
        )
        start_layer = resume_summary.completed_layers
        if start_layer >= args.max_layers:
            raise ValueError("resume checkpoint already contains all requested layers")
        total_logical_bits = resume_summary.total_logical_bits
        total_quantized_weights = resume_summary.total_quantized_weights
        previous_elapsed = float(result.get("elapsed_seconds", 0.0))
        run_started = time.time() - previous_elapsed
        resume_event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "from_layer": int(start_layer),
            "stream_dense_targets": bool(args.stream_dense_targets),
        }
        result.setdefault("resume_history", []).append(resume_event)
        checkpoint.setdefault("resume_history", []).append(resume_event)
        result.pop("completed", None)
        emit_progress(
            "resume_started",
            completed_layers=start_layer,
            next_layer=start_layer,
            stream_dense_targets=args.stream_dense_targets,
        )
        replay_completed_prefix(
            model,
            checkpoint,
            start_layer,
            train_inputs,
            validation_inputs,
            gptq_inputs,
            args,
            device,
        )
        emit_progress(
            "resume_prefix_replay_complete",
            completed_layers=start_layer,
            next_layer=start_layer,
        )
    else:
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment": (
                f"{Path(args.model).name.lower().replace('-', '_')}_"
                f"nowag_d{args.vector_dim}_vector_gptq_vector_gsq_prefix"
            ),
            "arguments": vars(args),
            "model_family": model_family,
            "data_shape": {
                "train_source_sequences": int(train_source_ids.shape[0]),
                "train_source_length": int(train_source_ids.shape[1]),
                "train_effective_sequences": int(train_ids.shape[0]),
                "train_effective_length": int(train_ids.shape[1]),
                "validation_source_sequences": int(validation_source_ids.shape[0]),
                "validation_source_length": int(
                    validation_source_ids.shape[1]
                    if validation_source_ids.ndim == 2
                    else 0
                ),
                "validation_effective_sequences": int(validation_ids.shape[0]),
                "validation_effective_length": int(
                    validation_ids.shape[1] if validation_ids.ndim == 2 else 0
                ),
                "gptq_sequences": int(gptq_ids.shape[0]),
                "gptq_sequence_length": int(gptq_ids.shape[1]),
            },
            "layers": [],
            "prefix_ppl": {},
        }
        checkpoint = {
            "metadata": {**vars(args), "model_family": model_family},
            "layers": {},
        }
        start_layer = 0
        total_logical_bits = 0
        total_quantized_weights = 0
        run_started = time.time()

    for layer_index in range(start_layer, args.max_layers):
        layer_started = time.time()
        emit_progress(
            "layer_started",
            layer=layer_index,
            total_layers=args.max_layers,
        )
        move_layer_for_quantization(
            model,
            layer_index,
            device,
            enabled=args.layerwise_cpu_offload,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        modules = get_linear_modules(model, layer_index)
        emit_progress("hessian_capture_started", layer=layer_index)
        if args.use_hidden_cache:
            hessians, capture_stats = collect_hessians_from_hidden(
                model, layer_index, modules, gptq_inputs, device
            )
        else:
            hessians, capture_stats = collect_hessian_diagonals(
                model,
                layer_index,
                modules,
                gptq_inputs,
                args.calib_sequences,
                args.calib_sequence_length,
                device,
            )
        emit_progress("hessian_capture_complete", layer=layer_index)
        dense_batch_size = (
            args.cache_microbatch_size
            if args.use_hidden_cache
            else args.train_batch_size
        )
        train_dense_outputs = (
            torch.empty(0)
            if args.stream_dense_targets
            else (
                collect_dense_block_outputs(
                    model,
                    layer_index,
                    modules,
                    train_inputs,
                    dense_batch_size,
                    device,
                    args.use_hidden_cache,
                )
                if train_inputs.shape[0]
                else torch.empty(0)
            )
        )
        if validation_inputs.shape[0] > 0:
            validation_dense_outputs = collect_dense_block_outputs(
                model,
                layer_index,
                modules,
                validation_inputs,
                args.cache_microbatch_size
                if args.use_hidden_cache
                else args.validation_batch_size,
                device,
                args.use_hidden_cache,
            )
            metric_ids = validation_inputs
            metric_outputs = validation_dense_outputs
            metric_batch_size = (
                args.cache_microbatch_size
                if args.use_hidden_cache
                else args.validation_batch_size
            )
        else:
            validation_dense_outputs = torch.empty(0)
            metric_ids = train_inputs
            metric_outputs = train_dense_outputs
            metric_batch_size = dense_batch_size

        sources = {}
        nowag_weights = {}
        gptq_weights = {}
        gptq_assignments = {}
        init_stats = {}
        gptq_wall_seconds = 0.0
        for linear_index, name in enumerate(LINEAR_NAMES):
            module = modules[name]
            weight = module.weight.detach().float()
            initialization_seed = args.seed + layer_index * 1_000 + linear_index
            seed_everything(initialization_seed)
            source, nowag_stats = run_full_nowag_initialization(
                weight,
                torch.diag(hessians[name]),
                args,
            )
            nowag_weight = source.reconstruct().detach().float()
            nowag_weights[name] = nowag_weight
            sources[name] = source
            init_stats[name] = {
                "initialization_seed": initialization_seed,
                **nowag_stats,
            }
        emit_progress("nowag_initialization_complete", layer=layer_index)

        if args.stage in ("gptq", "gsq"):
            emit_progress(
                "vector_gptq_started",
                layer=layer_index,
                concurrency=min(args.gptq_concurrency, len(LINEAR_NAMES)),
            )
            gptq_started = time.time()
            gptq_results = _run_vector_gptq_jobs(
                list(LINEAR_NAMES),
                modules,
                sources,
                hessians,
                args,
            )
            gptq_wall_seconds = time.time() - gptq_started
            for name in LINEAR_NAMES:
                refined_weight, refined_ids, gptq_stats = gptq_results[name]
                gptq_weights[name] = refined_weight
                gptq_assignments[name] = refined_ids
                init_stats[name].update(gptq_stats)
            emit_progress(
                "vector_gptq_complete",
                layer=layer_index,
                concurrency=min(args.gptq_concurrency, len(LINEAR_NAMES)),
                wall_seconds=gptq_wall_seconds,
            )
        else:
            for name in LINEAR_NAMES:
                gptq_weights[name] = nowag_weights[name]
                gptq_assignments[name] = sources[name].assignments.detach().long()
                init_stats[name].update(
                    {
                        "gptq_refinement_accepted": 0.0,
                        "skipped_by_stage": 1.0,
                    }
                )

        nowag_block_mse = dataset_block_mse(
            model,
            layer_index,
            modules,
            metric_ids,
            metric_outputs,
            nowag_weights,
            metric_batch_size,
            device,
            args.use_hidden_cache,
        )
        gptq_block_mse = dataset_block_mse(
            model,
            layer_index,
            modules,
            metric_ids,
            metric_outputs,
            gptq_weights,
            metric_batch_size,
            device,
            args.use_hidden_cache,
        )

        final_group_scales: Dict[str, torch.Tensor] = {}
        if args.stage == "nowag":
            final_weights = nowag_weights
            final_assignments = {
                name: sources[name].assignments.detach().long() for name in LINEAR_NAMES
            }
            gsq_stats = {"skipped_by_stage": 1.0}
        elif args.stage == "gptq":
            final_weights = gptq_weights
            final_assignments = gptq_assignments
            gsq_stats = {"skipped_by_stage": 1.0}
        else:
            del nowag_weights, gptq_weights
            torch.cuda.empty_cache()
            layer_args = args
            if layer_index < args.early_layers:
                layer_args = argparse.Namespace(**vars(args))
                layer_args.local_candidates = args.early_local_candidates
                layer_args.steps = args.early_steps
            if args.gsq_objective == "staged":
                (
                    final_weights,
                    final_assignments,
                    final_group_scales,
                    gsq_stats,
                ) = train_staged_block_gsq(
                    model,
                    layer_index,
                    modules,
                    train_inputs,
                    train_dense_outputs,
                    validation_inputs,
                    validation_dense_outputs,
                    sources,
                    hessians,
                    gptq_assignments,
                    layer_args,
                )
            else:
                final_weights, final_assignments, gsq_stats = train_block_gsq(
                    model,
                    layer_index,
                    modules,
                    train_inputs,
                    train_dense_outputs,
                    validation_inputs,
                    validation_dense_outputs,
                    sources,
                    hessians,
                    gptq_assignments,
                    layer_args,
                    args.use_hidden_cache,
                )
            if args.scale_finetune_steps > 0:
                if args.scale_finetune_mode == "group":
                    final_weights, final_group_scales, scale_stats = (
                        finetune_fixed_assignment_group_scales(
                            model,
                            layer_index,
                            modules,
                            train_inputs,
                            train_dense_outputs,
                            validation_inputs,
                            validation_dense_outputs,
                            sources,
                            final_weights,
                            final_assignments,
                            layer_args,
                        )
                    )
                else:
                    final_weights, scale_stats = finetune_fixed_assignment_row_scales(
                        model,
                        layer_index,
                        modules,
                        train_inputs,
                        train_dense_outputs,
                        validation_inputs,
                        validation_dense_outputs,
                        sources,
                        final_weights,
                        final_assignments,
                        layer_args,
                    )
                gsq_stats["scale_finetune"] = scale_stats
            emit_progress(
                "vector_gsq_complete",
                layer=layer_index,
                training_seconds=gsq_stats.get("training_seconds"),
            )

        layer_checkpoint = {}
        checkpoint_audit = {}
        for name in LINEAR_NAMES:
            module = modules[name]
            state, logical_bits = logical_state(
                sources[name],
                final_assignments[name],
                tuple(module.weight.shape),
                group_scale=final_group_scales.get(name),
                group_size=(
                    args.scale_group_size if name in final_group_scales else None
                ),
            )
            deployed_weight = reconstruct_logical_state(state, module.weight.device)
            target_weight = final_weights[name].float()
            relative_l2 = torch.linalg.vector_norm(
                deployed_weight - target_weight
            ) / torch.linalg.vector_norm(target_weight).clamp_min(1e-30)
            module.weight.data.copy_(deployed_weight.to(module.weight.dtype))
            reconstructed_model_weight = deployed_weight.to(module.weight.dtype)
            checkpoint_audit[name] = {
                "assignment_bits": int(state["assignment_bits"]),
                "used_codewords": int(
                    torch.unique(state["assignments"].long()).numel()
                ),
                "codebook_size": int(state["codebook"].shape[0]),
                "metadata_rounding_relative_l2": float(relative_l2),
                "model_reconstruction_exact": bool(
                    torch.equal(module.weight.detach(), reconstructed_model_weight)
                ),
                "model_reconstruction_max_abs": float(
                    (module.weight.detach() - reconstructed_model_weight).abs().max()
                ),
                "group_scale_count": int(
                    state["group_scale"].numel() if "group_scale" in state else 0
                ),
                "group_size": int(state["group_size"])
                if "group_size" in state
                else None,
            }
            layer_checkpoint[name] = state
            total_logical_bits += logical_bits
            total_quantized_weights += module.weight.numel()
        checkpoint["layers"][str(layer_index)] = layer_checkpoint

        layer_result = {
            "layer": layer_index,
            "capture": capture_stats,
            "initialization": init_stats,
            "initialization_policy": {
                "nowag_concurrency": 1,
                "gptq_concurrency": (
                    min(args.gptq_concurrency, len(LINEAR_NAMES))
                    if args.stage in ("gptq", "gsq")
                    else 0
                ),
                "gptq_wall_seconds": gptq_wall_seconds,
            },
            "block_mse_split": "validation"
            if validation_inputs.shape[0]
            else "training",
            "nowag_block_mse": float(nowag_block_mse),
            "gptq_block_mse": float(gptq_block_mse),
            "gsq": gsq_stats,
            "effective_gsq_policy": {
                "objective": args.gsq_objective,
                "local_candidates": int(
                    args.early_local_candidates
                    if layer_index < args.early_layers
                    else args.local_candidates
                ),
                "steps": int(
                    args.early_steps if layer_index < args.early_layers else args.steps
                ),
                "stage_steps_qk": int(args.stage_steps_qk or layer_args.steps),
                "stage_steps_vo": int(args.stage_steps_vo or layer_args.steps),
                "stage_steps_mlp": int(args.stage_steps_mlp or layer_args.steps),
                "staged_joint_group_scales": bool(args.staged_joint_group_scales),
                "staged_joint_scale_lr": float(args.staged_joint_scale_lr),
                "scale_finetune_steps": int(args.scale_finetune_steps),
                "scale_finetune_mode": args.scale_finetune_mode,
                "scale_group_size": int(args.scale_group_size),
                "block_polish_steps": int(args.block_polish_steps),
                "policy": ("early" if layer_index < args.early_layers else "default"),
            },
            "checkpoint_audit": checkpoint_audit,
            "layer_seconds": time.time() - layer_started,
        }
        result["layers"].append(layer_result)

        del train_dense_outputs, validation_dense_outputs
        if args.use_hidden_cache:
            if train_inputs.shape[0]:
                propagate_hidden_cache_in_place(
                    model,
                    layer_index,
                    modules,
                    train_inputs,
                    args.cache_microbatch_size,
                    device,
                )
            if validation_inputs.shape[0]:
                propagate_hidden_cache_in_place(
                    model,
                    layer_index,
                    modules,
                    validation_inputs,
                    args.cache_microbatch_size,
                    device,
                )
            propagate_hidden_cache_in_place(
                model,
                layer_index,
                modules,
                gptq_inputs,
                args.cache_microbatch_size,
                device,
            )

        depth = layer_index + 1
        if depth in depths:
            ppl = evaluate_wikitext2(
                model,
                tokenizer,
                args.ppl_seqlen,
                args.ppl_batch_size,
            )
            result["prefix_ppl"][str(depth)] = ppl

        offload_layer_after_quantization(
            model,
            layer_index,
            enabled=args.layerwise_cpu_offload,
        )
        if args.layerwise_cpu_offload:
            sources.clear()
            hessians.clear()
            gptq_assignments.clear()
            final_weights.clear()
            final_assignments.clear()
            if "gptq_results" in locals():
                gptq_results.clear()
            del (
                module,
                source,
                weight,
                nowag_weight,
                refined_weight,
                refined_ids,
                deployed_weight,
                target_weight,
                reconstructed_model_weight,
                relative_l2,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        result["logical_effective_bpp"] = total_logical_bits / max(
            total_quantized_weights, 1
        )
        result["elapsed_seconds"] = time.time() - run_started
        tmp_result = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_result.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_result, output_path)
        torch.save(checkpoint, checkpoint_path)
        emit_progress(
            "layer_complete",
            layer=layer_index,
            total_layers=args.max_layers,
            layer_seconds=layer_result["layer_seconds"],
            logical_effective_bpp=result["logical_effective_bpp"],
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["gpu_peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    result["completed"] = True
    tmp_result = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_result.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_result, output_path)
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
