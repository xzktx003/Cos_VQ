"""Joint block-level Vector-GSQ after full NoWag d=6 initialization.

All seven Qwen3 Transformer-block Linear weights are independently initialized
with the original NoWag d=6, 2-bit-per-weight, K=4096 weighted VQ.  Their local
codeword probabilities are then optimized jointly against the dense block
output.  This is an experiment-only path and does not alter the established
scalar GSQ implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM

from experiments.nowag_d6.diagnose_vector_gsq_gradients import (
    LocalVectorGumbelQuantizer,
    Schedule,
    build_local_candidates,
    gradient_metrics,
    initialize_logits,
    normalized_vectors_and_weights,
    probability_metrics,
    run_full_nowag_initialization,
    seed_everything,
    vector_codebook_aware_gptq,
)
from experiments.nowag_d6.logical_state import (
    LINEAR_NAMES,
    get_layer,
    get_linear_modules,
)


class _StopAfterBlock(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nowag-repo",
        default=str(Path(__file__).resolve().parents[2] / "third_party" / "NoWag"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--calib-sequences", type=int, default=8)
    parser.add_argument("--calib-sequence-length", type=int, default=512)
    parser.add_argument("--train-sequence-length", type=int, default=128)
    parser.add_argument("--vector-dim", type=int, default=6)
    parser.add_argument("--bits-per-weight", type=float, default=2.0)
    parser.add_argument("--lloyd-iters", type=int, default=100)
    parser.add_argument("--local-candidates", type=int, default=8)
    parser.add_argument("--candidate-chunk", type=int, default=4096)
    parser.add_argument("--gptq-percdamp", type=float, default=0.1)
    parser.add_argument("--gptq-row-chunk", type=int, default=256)
    parser.add_argument("--gptq-candidates", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--schedule", choices=("original", "gentle"), default="original")
    parser.add_argument("--half-kmeans", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def collect_hessian_diagonals(
    model: nn.Module,
    layer: int,
    modules: Dict[str, nn.Linear],
    token_ids: torch.Tensor,
    sequences: int,
    sequence_length: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    sums = {
        name: torch.zeros(
            module.in_features,
            module.in_features,
            device=device,
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

    block = get_layer(model, layer)
    block_handle = block.register_forward_hook(
        lambda _module, _inputs, _output: (_ for _ in ()).throw(_StopAfterBlock())
    )
    started = time.time()
    try:
        for index in range(sequences):
            ids = token_ids[index : index + 1, :sequence_length].to(device)
            try:
                model(input_ids=ids, use_cache=False)
            except _StopAfterBlock:
                pass
    finally:
        block_handle.remove()
        for handle in handles:
            handle.remove()

    hessians = {
        name: sums[name] / max(counts[name], 1)
        for name in modules
    }
    elapsed = time.time() - started
    stats = {
        name: {
            "rows": float(counts[name]),
            "diag_min": float(torch.diag(hessians[name]).min()),
            "diag_max": float(torch.diag(hessians[name]).max()),
            "diag_mean": float(torch.diag(hessians[name]).mean()),
            "shared_capture_seconds": elapsed,
        }
        for name in modules
    }
    return hessians, stats


@contextlib.contextmanager
def patched_linear_weights(
    modules: Dict[str, nn.Linear],
    weights: Dict[str, torch.Tensor] | None,
):
    originals = {}
    if weights is not None:
        for name, module in modules.items():
            if name not in weights:
                continue
            originals[name] = module.forward
            weight = weights[name]

            def make_forward(current_weight, current_module):
                def forward(_self, x):
                    return F.linear(x, current_weight.to(x.dtype), current_module.bias)
                return forward

            module.forward = make_forward(weight, module).__get__(module, nn.Linear)
    try:
        yield
    finally:
        for name, module in modules.items():
            if name in originals:
                module.forward = originals[name]


def run_to_block_output(
    model: nn.Module,
    layer: int,
    modules: Dict[str, nn.Linear],
    input_ids: torch.Tensor,
    weights: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    captured: List[torch.Tensor] = []
    block = get_layer(model, layer)

    def output_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            captured.append(output[0])
        else:
            captured.append(output)
        raise _StopAfterBlock()

    handle = block.register_forward_hook(output_hook)
    try:
        with patched_linear_weights(modules, weights):
            try:
                model(input_ids=input_ids, use_cache=False)
            except _StopAfterBlock:
                pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Block output hook did not run")
    return captured[0]


def aggregate_probability_metrics(
    quantizers: Dict[str, LocalVectorGumbelQuantizer],
    temperature: float,
    kappa: float,
) -> Dict[str, float]:
    weighted = {}
    total = 0
    for quantizer in quantizers.values():
        metrics = probability_metrics(quantizer.quant_logits, temperature, kappa)
        count = quantizer.quant_logits.shape[0]
        total += count
        for key, value in metrics.items():
            weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / max(total, 1) for key, value in weighted.items()}


def aggregate_gradient_metrics(
    quantizers: Dict[str, LocalVectorGumbelQuantizer],
) -> Dict[str, float]:
    total_elements = 0
    square_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    lt12 = 0
    lt10 = 0
    nonfinite = 0
    for quantizer in quantizers.values():
        grad = quantizer.quant_logits.grad.detach().float()
        total_elements += grad.numel()
        square_sum += float(grad.square().sum())
        abs_sum += float(grad.abs().sum())
        max_abs = max(max_abs, float(grad.abs().max()))
        lt12 += int((grad.abs() < 1e-12).sum())
        lt10 += int((grad.abs() < 1e-10).sum())
        nonfinite += int((~torch.isfinite(grad)).sum())
    return {
        "grad_mean_abs": abs_sum / max(total_elements, 1),
        "grad_rms": math.sqrt(square_sum / max(total_elements, 1)),
        "grad_max_abs": max_abs,
        "grad_lt_1e_12_rate": lt12 / max(total_elements, 1),
        "grad_lt_1e_10_rate": lt10 / max(total_elements, 1),
        "grad_nonfinite_rate": nonfinite / max(total_elements, 1),
    }


@torch.no_grad()
def hard_weights_and_switch_rate(
    quantizers: Dict[str, LocalVectorGumbelQuantizer],
    initial_assignments: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
    weights = {}
    switched = 0
    total = 0
    per_linear = {}
    for name, quantizer in quantizers.items():
        weight, ids = quantizer.hard_weight()
        weights[name] = weight
        rate = float((ids != initial_assignments[name]).float().mean())
        per_linear[name] = rate
        switched += int((ids != initial_assignments[name]).sum())
        total += ids.numel()
    return weights, switched / max(total, 1), per_linear


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    token_ids = payload["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules = get_linear_modules(model, args.layer)
    hessians, capture_stats = collect_hessian_diagonals(
        model,
        args.layer,
        modules,
        token_ids,
        args.calib_sequences,
        args.calib_sequence_length,
        device,
    )
    train_ids = token_ids[:1, : args.train_sequence_length].to(device)
    with torch.inference_mode():
        dense_output = run_to_block_output(
            model, args.layer, modules, train_ids, weights=None
        ).detach()

    source_quantizers = {}
    pre_gptq_weights = {}
    candidate_ids = {}
    candidate_distances = {}
    original_columns = {}
    initial_assignments = {}
    initialization = {}

    for name, module in modules.items():
        weight = module.weight.detach().float()
        source, init_stats = run_full_nowag_initialization(
            weight,
            torch.diag(hessians[name]),
            args,
        )
        pre_gptq_weights[name] = source.reconstruct().detach().float()
        refined_weight, refined_assignments, gptq_stats = vector_codebook_aware_gptq(
            source,
            weight,
            hessians[name],
            args.vector_dim,
            percdamp=args.gptq_percdamp,
            row_chunk=args.gptq_row_chunk,
            candidate_count=args.gptq_candidates,
        )
        vectors, vector_weights, columns = normalized_vectors_and_weights(
            source,
            weight,
            torch.diag(hessians[name]),
            args.vector_dim,
        )
        ids, distances, candidate_stats = build_local_candidates(
            vectors,
            vector_weights,
            source.codebook.detach().float(),
            refined_assignments.detach().long(),
            args.local_candidates,
            args.candidate_chunk,
        )
        source_quantizers[name] = source
        candidate_ids[name] = ids
        candidate_distances[name] = distances
        original_columns[name] = columns
        initial_assignments[name] = refined_assignments.detach().long()
        initialization[name] = {
            **init_stats,
            **gptq_stats,
            **candidate_stats,
            "shape": list(weight.shape),
            "refined_weight_relative_l2": float(
                (refined_weight - weight).norm() / weight.norm().clamp_min(1e-12)
            ),
        }

    quantizers = {}
    optimizer_parameters = []
    for index, name in enumerate(LINEAR_NAMES):
        logits = initialize_logits(
            candidate_distances[name],
            args.std,
            args.strength,
            args.seed + index,
        )
        quantizer = LocalVectorGumbelQuantizer(
            source_quantizers[name],
            candidate_ids[name],
            logits,
            original_columns[name],
        ).to(device)
        quantizers[name] = quantizer
        optimizer_parameters.append(quantizer.quant_logits)

    if args.schedule == "original":
        schedule = Schedule("gsq_original", 2.0, 0.05, 100.0, 500.0)
    else:
        schedule = Schedule("gradient_friendly_control", 2.0, 0.2, 25.0, 100.0)

    optimizer = Lion(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * 0.1,
    )

    initial_hard_weights, _, _ = hard_weights_and_switch_rate(
        quantizers, initial_assignments
    )
    with torch.inference_mode():
        pre_gptq_output = run_to_block_output(
            model, args.layer, modules, train_ids, pre_gptq_weights
        )
        pre_gptq_hard_loss = F.mse_loss(
            pre_gptq_output.float(), dense_output.float()
        )
        initial_hard_output = run_to_block_output(
            model, args.layer, modules, train_ids, initial_hard_weights
        )
        initial_hard_loss = F.mse_loss(initial_hard_output.float(), dense_output.float())

    checkpoints = {0, 1, 5, 10, 25, 50, 100, 200, args.steps - 1}
    trace = []
    started = time.time()
    for step in range(args.steps):
        t = step / max(args.steps - 1, 1)
        temperature = schedule.temperature_start + (
            schedule.temperature_end - schedule.temperature_start
        ) * t
        kappa = schedule.kappa_start + (schedule.kappa_end - schedule.kappa_start) * t

        optimizer.zero_grad(set_to_none=True)
        soft_weights = {
            name: quantizer(temperature, kappa, add_gumbel=True)[0]
            for name, quantizer in quantizers.items()
        }
        output = run_to_block_output(
            model, args.layer, modules, train_ids, soft_weights
        )
        loss = F.mse_loss(output.float(), dense_output.float())
        loss.backward()

        if step in checkpoints:
            with torch.no_grad():
                hard_weights, switch_rate, per_linear_switch = hard_weights_and_switch_rate(
                    quantizers, initial_assignments
                )
                hard_output = run_to_block_output(
                    model, args.layer, modules, train_ids, hard_weights
                )
                hard_loss = F.mse_loss(hard_output.float(), dense_output.float())
            trace.append({
                "step": float(step),
                "temperature": float(temperature),
                "kappa": float(kappa),
                "soft_block_mse": float(loss.detach()),
                "hard_block_mse": float(hard_loss),
                "assignment_switch_rate": float(switch_rate),
                "per_linear_switch_rate": per_linear_switch,
                **aggregate_probability_metrics(quantizers, temperature, kappa),
                **aggregate_gradient_metrics(quantizers),
            })

        optimizer.step()
        scheduler.step()
        del soft_weights, output, loss

    with torch.inference_mode():
        final_hard_weights, switch_rate, per_linear_switch = hard_weights_and_switch_rate(
            quantizers, initial_assignments
        )
        final_hard_output = run_to_block_output(
            model, args.layer, modules, train_ids, final_hard_weights
        )
        final_hard_loss = F.mse_loss(final_hard_output.float(), dense_output.float())

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "full_nowag_d6_all_block_linears_then_joint_local_vector_gsq",
        "arguments": vars(args),
        "schedule": asdict(schedule),
        "hessian_capture": capture_stats,
        "initialization": initialization,
        "nowag_initial_hard_block_mse": float(pre_gptq_hard_loss),
        "initial_hard_block_mse": float(initial_hard_loss),
        "final_hard_block_mse": float(final_hard_loss),
        "hard_block_mse_relative_change": float(
            (final_hard_loss - initial_hard_loss) / initial_hard_loss.clamp_min(1e-30)
        ),
        "final_assignment_switch_rate": float(switch_rate),
        "final_per_linear_switch_rate": per_linear_switch,
        "training_seconds": time.time() - started,
        "trace": trace,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
