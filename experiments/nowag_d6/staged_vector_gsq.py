"""Structure-aware objectives for staged Vector-GSQ optimization.

This module mirrors the useful optimization granularity from scalar GSQ while
keeping Vector-GSQ's codebook and logical-checkpoint representation intact.
The discrete phases are Q, K, V/O, and MLP.  A separate continuous phase can
either fold output-row scales into ``row_norm`` or learn deployable input-group
scales while keeping assignments and the vector codebook fixed.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

from experiments.nowag_d6.diagnose_block_vector_gsq import patched_linear_weights


Q_PROJ = "self_attn.q_proj"
K_PROJ = "self_attn.k_proj"
V_PROJ = "self_attn.v_proj"
O_PROJ = "self_attn.o_proj"
MLP_NAMES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@dataclass(frozen=True)
class StagedPhaseSpec:
    name: str
    active_names: tuple[str, ...]
    objective: str
    frozen_names: tuple[str, ...]


def staged_phase_specs() -> tuple[StagedPhaseSpec, ...]:
    """Return the canonical structure-aware phase order."""
    return (
        StagedPhaseSpec("q_proj", (Q_PROJ,), "linear_q", ()),
        StagedPhaseSpec("k_proj", (K_PROJ,), "linear_k", (Q_PROJ,)),
        StagedPhaseSpec(
            "v_o",
            (V_PROJ, O_PROJ),
            "attention",
            (Q_PROJ, K_PROJ),
        ),
        StagedPhaseSpec(
            "mlp",
            MLP_NAMES,
            "block",
            (Q_PROJ, K_PROJ, V_PROJ, O_PROJ),
        ),
    )


def validate_block_refinement_contract(
    staged_freeze_qk: bool,
    block_refinement_steps: int,
) -> None:
    """Reject a refinement mode that would silently unfreeze Q/K."""
    if staged_freeze_qk and block_refinement_steps > 0:
        raise ValueError(
            "block refinement currently updates every Linear and cannot be "
            "combined with staged-freeze-qk"
        )


def _position_inputs(model, block, hidden_states: torch.Tensor) -> dict[str, object]:
    position_ids = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(hidden_states.shape[0], -1)
    kwargs: dict[str, object] = {
        "attention_mask": None,
        "position_ids": position_ids,
        "use_cache": False,
    }
    attention = block.self_attn
    if "position_embeddings" in inspect.signature(attention.forward).parameters:
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is not None:
            kwargs["position_embeddings"] = rotary(hidden_states, position_ids)
    return kwargs


def _first_output(output):
    return output[0] if isinstance(output, tuple) else output


def run_hidden_stage_output(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor] | None,
    objective: str,
) -> torch.Tensor:
    """Run one staged objective from a cached Transformer-block input."""
    block = model.model.layers[layer_index]
    normalized = block.input_layernorm(hidden_states)
    if objective == "linear_q":
        with patched_linear_weights(modules, weights):
            return modules[Q_PROJ](normalized)
    if objective == "linear_k":
        with patched_linear_weights(modules, weights):
            return modules[K_PROJ](normalized)
    if objective not in {"attention", "block"}:
        raise ValueError(f"Unsupported staged objective: {objective}")

    kwargs = _position_inputs(model, block, normalized)
    with patched_linear_weights(modules, weights):
        attention_output = _first_output(block.self_attn(normalized, **kwargs))
        if objective == "attention":
            return attention_output
        mlp_input = hidden_states + attention_output
        mlp_output = block.mlp(block.post_attention_layernorm(mlp_input))
        return mlp_input + mlp_output


@torch.inference_mode()
def collect_hidden_stage_outputs(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor] | None,
    objective: str,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, hidden_states.shape[0], batch_size):
        batch = hidden_states[start : start + batch_size].to(
            next(iter(modules.values())).weight.device
        )
        output = run_hidden_stage_output(
            model,
            layer_index,
            modules,
            batch,
            weights,
            objective,
        )
        outputs.append(output.detach().to("cpu", torch.bfloat16))
    return torch.cat(outputs, dim=0)


def collect_dense_stage_targets(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    dense_block_outputs: torch.Tensor,
    objective: str,
    batch_size: int,
) -> torch.Tensor:
    """Return the dense teacher for one structure-aware training phase.

    The official GSQ objective always compares a quantized candidate with the
    original dense network.  In particular, V/O must be allowed to compensate
    Q/K error and the MLP must be allowed to compensate attention error.  A
    teacher produced with already-selected quantized prefix weights would
    remove precisely that compensation signal.
    """
    if objective == "block":
        if dense_block_outputs.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "dense block targets and hidden states must have the same "
                "number of sequences"
            )
        return dense_block_outputs
    if objective not in {"linear_q", "linear_k", "attention"}:
        raise ValueError(f"Unsupported staged objective: {objective}")
    return collect_hidden_stage_outputs(
        model,
        layer_index,
        modules,
        hidden_states,
        None,
        objective,
        batch_size,
    )


@torch.inference_mode()
def hidden_stage_mse(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    dense_outputs: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    objective: str,
    batch_size: int,
) -> float:
    squared_error = 0.0
    elements = 0
    device = next(iter(modules.values())).weight.device
    for start in range(0, hidden_states.shape[0], batch_size):
        end = min(start + batch_size, hidden_states.shape[0])
        batch = hidden_states[start:end].to(device)
        output = run_hidden_stage_output(
            model,
            layer_index,
            modules,
            batch,
            weights,
            objective,
        ).float()
        target = dense_outputs[start:end].to(output.device, torch.float32)
        squared_error += float((output - target).square().sum())
        elements += output.numel()
    return squared_error / max(elements, 1)


@contextlib.contextmanager
def patched_linear_row_scales(
    modules,
    weights: Mapping[str, torch.Tensor],
    row_scales: Mapping[str, torch.Tensor],
):
    """Patch fixed weights with differentiable output-row scale corrections."""
    originals = {}
    for name, weight in weights.items():
        module = modules[name]
        originals[name] = module.forward
        scale = row_scales.get(name)
        if scale is None:
            scale = torch.ones(
                module.out_features,
                device=weight.device,
                dtype=torch.float32,
            )

        def make_forward(current_weight, current_scale, current_module):
            def forward(_self, inputs):
                contribution = F.linear(inputs, current_weight.to(inputs.dtype), None)
                # Keep the continuous scale in FP32 until the final activation
                # cast. Casting a near-one scale to BF16 first makes practical
                # learning rates indistinguishable from exactly one.
                output = (
                    contribution.float() * current_scale.float()
                ).to(inputs.dtype)
                if current_module.bias is not None:
                    output = output + current_module.bias.to(output.dtype)
                return output

            return forward

        module.forward = make_forward(weight, scale, module).__get__(
            module, torch.nn.Linear
        )
    try:
        yield
    finally:
        for name, original in originals.items():
            modules[name].forward = original


def run_hidden_block_with_row_scales(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    row_scales: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    block = model.model.layers[layer_index]
    position_ids = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(hidden_states.shape[0], -1)
    kwargs: dict[str, object] = {
        "attention_mask": None,
        "position_ids": position_ids,
        "use_cache": False,
    }
    if "position_embeddings" in inspect.signature(block.forward).parameters:
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is not None:
            kwargs["position_embeddings"] = rotary(hidden_states, position_ids)
    with patched_linear_row_scales(modules, weights, row_scales):
        return _first_output(block(hidden_states, **kwargs))


def expand_group_scales(
    group_scales: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Expand ``[out_features, groups]`` scales to a weight-shaped tensor."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if group_scales.ndim != 2:
        raise ValueError("group_scales must be two-dimensional")
    expected_groups = (in_features + group_size - 1) // group_size
    if group_scales.shape[1] != expected_groups:
        raise ValueError(
            f"group_scales has {group_scales.shape[1]} groups, expected "
            f"{expected_groups} for in_features={in_features}, group_size={group_size}"
        )
    return group_scales.repeat_interleave(group_size, dim=1)[:, :in_features]


def apply_group_scales(
    weights: Mapping[str, torch.Tensor],
    modules: Mapping[str, torch.nn.Linear],
    group_scales: Mapping[str, torch.Tensor],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """Multiply selected weights by differentiable input-group scales.

    Names absent from ``group_scales`` are returned unchanged.  Keeping this
    operation separate from module patching lets staged Vector-GSQ optimize a
    phase's discrete assignments and continuous scales in the same graph.
    """
    scaled = {}
    for name, weight in weights.items():
        scale = group_scales.get(name)
        if scale is None:
            scaled[name] = weight
            continue
        if name not in modules:
            raise KeyError(f"Missing Linear module for {name}")
        module = modules[name]
        if scale.shape[0] != module.out_features:
            raise ValueError(
                f"{name} group scale has {scale.shape[0]} rows, expected "
                f"{module.out_features}"
            )
        expanded = expand_group_scales(
            scale,
            module.in_features,
            group_size,
        )
        scaled[name] = weight.float() * expanded.float()
    return scaled


def joint_block_refinement_loss(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    dense_outputs: torch.Tensor,
    quantizers: Mapping[str, torch.nn.Module],
    log_group_scales: Mapping[str, torch.Tensor],
    temperature: float,
    kappa: float,
    group_size: int,
    *,
    add_gumbel: bool = True,
) -> torch.Tensor:
    """Differentiate a dense-reference block loss through all active states."""
    soft_weights = {
        name: quantizer(temperature, kappa, add_gumbel=add_gumbel)[0]
        for name, quantizer in quantizers.items()
    }
    soft_weights = apply_group_scales(
        soft_weights,
        modules,
        {name: value.exp() for name, value in log_group_scales.items()},
        group_size,
    )
    output = run_hidden_stage_output(
        model,
        layer_index,
        modules,
        hidden_states,
        soft_weights,
        "block",
    )
    return F.mse_loss(output.float(), dense_outputs.to(output.device, torch.float32))


@dataclass(frozen=True)
class JointHardState:
    """Atomic assignment/scale checkpoint selected by one hard block loss."""

    loss: float
    assignments: Dict[str, torch.Tensor]
    log_group_scales: Dict[str, torch.Tensor]


def snapshot_joint_hard_state(
    loss: float,
    assignments: Mapping[str, torch.Tensor],
    log_group_scales: Mapping[str, torch.Tensor],
) -> JointHardState:
    return JointHardState(
        loss=float(loss),
        assignments={
            name: value.detach().clone() for name, value in assignments.items()
        },
        log_group_scales={
            name: value.detach().clone()
            for name, value in log_group_scales.items()
        },
    )


def select_better_joint_snapshot(
    current: JointHardState,
    loss: float,
    assignments: Mapping[str, torch.Tensor],
    log_group_scales: Mapping[str, torch.Tensor],
) -> JointHardState:
    """Atomically select a strictly better finite hard deployment candidate."""
    candidate_loss = float(loss)
    if not torch.isfinite(torch.tensor(candidate_loss)) or candidate_loss >= current.loss:
        return current
    return snapshot_joint_hard_state(
        candidate_loss,
        assignments,
        log_group_scales,
    )


@contextlib.contextmanager
def patched_linear_group_scales(
    modules,
    weights: Mapping[str, torch.Tensor],
    group_scales: Mapping[str, torch.Tensor],
    group_size: int,
):
    """Patch fixed weights with differentiable per-row input-group scales."""
    originals = {}
    for name, weight in weights.items():
        module = modules[name]
        scale = group_scales.get(name)
        if scale is None:
            expanded = torch.ones(
                module.out_features,
                module.in_features,
                device=weight.device,
                dtype=torch.float32,
            )
        else:
            if scale.shape[0] != module.out_features:
                raise ValueError(
                    f"{name} group scale has {scale.shape[0]} rows, expected "
                    f"{module.out_features}"
                )
            expanded = expand_group_scales(scale, module.in_features, group_size)
        originals[name] = module.forward

        def make_forward(current_weight, current_scale, current_module):
            def forward(_self, inputs):
                # The cast mirrors deployed BF16 weights. Autograd still flows
                # through the cast, so small Lion steps accumulate until they
                # cross an actually representable checkpoint value.
                scaled_weight = (
                    current_weight.float() * current_scale.float()
                ).to(inputs.dtype)
                return F.linear(inputs, scaled_weight, current_module.bias)

            return forward

        module.forward = make_forward(weight, expanded, module).__get__(
            module, torch.nn.Linear
        )
    try:
        yield
    finally:
        for name, original in originals.items():
            modules[name].forward = original


def run_hidden_block_with_group_scales(
    model,
    layer_index: int,
    modules,
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    group_scales: Mapping[str, torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    """Run one cached block with fixed codes and deployable group scales."""
    block = model.model.layers[layer_index]
    position_ids = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(hidden_states.shape[0], -1)
    kwargs: dict[str, object] = {
        "attention_mask": None,
        "position_ids": position_ids,
        "use_cache": False,
    }
    if "position_embeddings" in inspect.signature(block.forward).parameters:
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is not None:
            kwargs["position_embeddings"] = rotary(hidden_states, position_ids)
    with patched_linear_group_scales(
        modules,
        weights,
        group_scales,
        group_size,
    ):
        return _first_output(block(hidden_states, **kwargs))


def fold_row_scales_(
    sources: Mapping[str, object],
    weights: Dict[str, torch.Tensor],
    row_scales: Mapping[str, torch.Tensor],
) -> None:
    """Fold selected row scales into existing logical checkpoint factors."""
    with torch.no_grad():
        for name, scale in row_scales.items():
            factor = scale.detach().reshape(-1)
            row_norm = sources[name].normalizer.norms[1]
            if row_norm.numel() != factor.numel():
                raise ValueError(
                    f"{name} row scale has {factor.numel()} values, expected "
                    f"{row_norm.numel()}"
                )
            row_norm.mul_(factor.to(row_norm).reshape_as(row_norm))
            weights[name].mul_(factor.to(weights[name])[:, None])


def merge_selected_weights(
    quantizers: Mapping[str, object],
    assignments: Mapping[str, torch.Tensor],
    names: Iterable[str],
) -> Dict[str, torch.Tensor]:
    return {
        name: quantizers[name].weight_from_global_ids(assignments[name])
        for name in names
    }


def anchor_quantizer_logits_(
    quantizer,
    selected_ids: torch.Tensor,
    margin: float,
) -> None:
    """Make selected global IDs the hard state without discarding learned logits."""
    if margin <= 0:
        raise ValueError("anchor margin must be positive")
    selected = selected_ids.to(quantizer.candidate_ids.device).reshape(-1)
    if selected.numel() != quantizer.candidate_ids.shape[0]:
        raise ValueError("selected assignment count does not match candidate rows")
    matches = quantizer.candidate_ids.eq(selected[:, None])
    if not bool(matches.any(dim=1).all()):
        raise ValueError("selected assignment is absent from a local candidate set")
    positions = matches.to(torch.int64).argmax(dim=1)
    rows = torch.arange(selected.numel(), device=selected.device)
    with torch.no_grad():
        selected_logits = quantizer.quant_logits[rows, positions]
        maximum = quantizer.quant_logits.amax(dim=1)
        quantizer.quant_logits[rows, positions] = torch.maximum(
            selected_logits,
            maximum + float(margin),
        )
