"""Behavioral contract for staged, activation-aware Vector-GSQ training."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
from torch import nn

if "lion_pytorch" not in sys.modules:
    sys.modules["lion_pytorch"] = types.SimpleNamespace(Lion=object)

from experiments.nowag_d6.diagnose_block_vector_gsq import patched_linear_weights
from experiments.nowag_d6.staged_vector_gsq import (
    anchor_quantizer_logits_,
    fold_row_scales_,
    patched_linear_row_scales,
    run_hidden_block_with_row_scales,
    run_hidden_stage_output,
    staged_phase_specs,
)


def test_staged_phase_specs_follow_the_gsq_network_structure() -> None:
    specs = staged_phase_specs()

    assert [spec.name for spec in specs] == ["q_proj", "k_proj", "v_o", "mlp"]
    assert [tuple(spec.active_names) for spec in specs] == [
        ("self_attn.q_proj",),
        ("self_attn.k_proj",),
        ("self_attn.v_proj", "self_attn.o_proj"),
        ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    ]
    assert [spec.objective for spec in specs] == [
        "linear_q",
        "linear_k",
        "attention",
        "block",
    ]


def test_block_refinement_rejects_silently_unfreezing_qk() -> None:
    from experiments.nowag_d6.staged_vector_gsq import (
        validate_block_refinement_contract,
    )

    validate_block_refinement_contract(False, 1280)
    validate_block_refinement_contract(True, 0)
    with pytest.raises(ValueError, match="cannot be combined with staged-freeze-qk"):
        validate_block_refinement_contract(True, 1)


def test_patched_linear_weights_accepts_partial_mapping_and_restores_forwards() -> None:
    q_proj = nn.Linear(2, 2, bias=False)
    k_proj = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        q_proj.weight.copy_(torch.eye(2))
        k_proj.weight.copy_(2.0 * torch.eye(2))

    modules = {
        "self_attn.q_proj": q_proj,
        "self_attn.k_proj": k_proj,
    }
    inputs = torch.tensor([[1.0, 3.0]])
    q_before = q_proj(inputs).detach().clone()
    k_before = k_proj(inputs).detach().clone()
    original_q_forward = q_proj.forward
    original_k_forward = k_proj.forward

    with patched_linear_weights(
        modules,
        {"self_attn.q_proj": 4.0 * torch.eye(2)},
    ):
        torch.testing.assert_close(q_proj(inputs), 4.0 * inputs)
        torch.testing.assert_close(k_proj(inputs), k_before)
        assert q_proj.forward != original_q_forward
        assert k_proj.forward == original_k_forward

    torch.testing.assert_close(q_proj(inputs), q_before)
    torch.testing.assert_close(k_proj(inputs), k_before)
    assert q_proj.forward == original_q_forward
    assert k_proj.forward == original_k_forward


class _FakeNormalizer:
    def __init__(self, col_norm: torch.Tensor, row_norm: torch.Tensor):
        self.norms = [col_norm, row_norm]


class _FakeSource:
    def __init__(self) -> None:
        self.assignments = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.codebook = torch.tensor(
            [[1.0, -1.0], [0.5, -0.5]], dtype=torch.float64
        )
        self.normalizer = _FakeNormalizer(
            col_norm=torch.tensor([3.0, 5.0], dtype=torch.float64),
            row_norm=torch.tensor([2.0, 4.0], dtype=torch.float64),
        )


def test_fold_row_scales_preserves_discrete_state_and_folds_exactly() -> None:
    source = _FakeSource()
    sources = {"mlp.down_proj": source}
    weights = {
        "mlp.down_proj": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        )
    }
    scales = {
        "mlp.down_proj": torch.tensor([2.0, 0.5], dtype=torch.float64)
    }
    assignments_before = source.assignments.clone()
    codebook_before = source.codebook.clone()
    col_norm_before = source.normalizer.norms[0].clone()

    fold_row_scales_(sources, weights, scales)

    torch.testing.assert_close(
        weights["mlp.down_proj"],
        torch.tensor([[2.0, 4.0], [1.5, 2.0]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        source.normalizer.norms[1],
        torch.tensor([4.0, 2.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(source.normalizer.norms[0], col_norm_before)
    assert torch.equal(source.assignments, assignments_before)
    assert torch.equal(source.codebook, codebook_before)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(5, 6, bias=False)
        self.k_proj = nn.Linear(5, 4, bias=False)
        self.v_proj = nn.Linear(5, 4, bias=False)
        self.o_proj = nn.Linear(4, 5, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        # Keep Q/K in the output dependency graph while retaining GQA-like
        # unequal projection widths in this deterministic CPU fixture.
        queries = self.q_proj(hidden_states)[..., :4]
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        return self.o_proj(values + 0.01 * queries * keys), None


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(5, 7, bias=False)
        self.up_proj = nn.Linear(5, 7, bias=False)
        self.down_proj = nn.Linear(7, 5, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.sigmoid(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class _TinyLlamaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _TinyAttention()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _TinyMlp()

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, _ = self.self_attn(
            self.input_layernorm(hidden_states), **kwargs
        )
        attention_residual = hidden_states + attention_output
        block_output = attention_residual + self.mlp(
            self.post_attention_layernorm(attention_residual)
        )
        return (block_output,)


class _TinyLlama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([_TinyLlamaBlock()])
        self.model = backbone


def _tiny_linear_modules(model: _TinyLlama) -> dict[str, nn.Linear]:
    block = model.model.layers[0]
    return {
        "self_attn.q_proj": block.self_attn.q_proj,
        "self_attn.k_proj": block.self_attn.k_proj,
        "self_attn.v_proj": block.self_attn.v_proj,
        "self_attn.o_proj": block.self_attn.o_proj,
        "mlp.gate_proj": block.mlp.gate_proj,
        "mlp.up_proj": block.mlp.up_proj,
        "mlp.down_proj": block.mlp.down_proj,
    }


def test_hidden_stage_output_returns_each_requested_llama_objective_shape() -> None:
    torch.manual_seed(0)
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    hidden_states = torch.randn(2, 3, 5)

    expected_shapes = {
        "linear_q": (2, 3, 6),
        "linear_k": (2, 3, 4),
        "attention": (2, 3, 5),
        "block": (2, 3, 5),
    }
    for objective, expected_shape in expected_shapes.items():
        output = run_hidden_stage_output(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            weights=None,
            objective=objective,
        )
        assert tuple(output.shape) == expected_shape
        assert torch.isfinite(output).all()


def test_fixed_weight_row_scales_receive_gradients_and_preserve_unit_output() -> None:
    torch.manual_seed(1)
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    hidden_states = torch.randn(2, 3, 5)
    weights = {
        name: module.weight.detach().clone() for name, module in modules.items()
    }
    scales = {
        name: nn.Parameter(torch.ones(module.out_features))
        for name, module in modules.items()
    }

    dense_output = run_hidden_stage_output(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        weights=weights,
        objective="block",
    )
    scaled_output = run_hidden_block_with_row_scales(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        weights=weights,
        row_scales=scales,
    )
    torch.testing.assert_close(scaled_output, dense_output)

    scaled_output.square().mean().backward()
    assert all(scale.grad is not None for scale in scales.values())
    assert any(torch.count_nonzero(scale.grad).item() for scale in scales.values())

    original_forwards = {name: module.forward for name, module in modules.items()}
    with patched_linear_row_scales(modules, weights, scales):
        assert any(
            module.forward != original_forwards[name]
            for name, module in modules.items()
        )
    assert all(
        module.forward == original_forwards[name]
        for name, module in modules.items()
    )


def test_row_scale_change_is_observable_with_bfloat16_activations() -> None:
    linear = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    modules = {"self_attn.q_proj": linear}
    weights = {"self_attn.q_proj": linear.weight.detach().float().clone()}
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    baseline = linear(inputs).detach().clone()
    scale = nn.Parameter(torch.full((3,), 1.01, dtype=torch.float32))

    with patched_linear_row_scales(
        modules,
        weights,
        {"self_attn.q_proj": scale},
    ):
        output = linear(inputs)

    assert output.dtype == torch.bfloat16
    assert not torch.equal(output, baseline)
    output.float().sum().backward()
    assert scale.grad is not None
    assert torch.count_nonzero(scale.grad).item() == scale.numel()


def test_anchor_quantizer_logits_restores_selected_hard_assignments() -> None:
    quantizer = types.SimpleNamespace(
        candidate_ids=torch.tensor([[4, 7, 9], [2, 5, 8]]),
        quant_logits=nn.Parameter(
            torch.tensor([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0]])
        ),
    )

    anchor_quantizer_logits_(quantizer, torch.tensor([9, 2]), margin=0.25)

    local_ids = quantizer.quant_logits.argmax(dim=1)
    rows = torch.arange(2)
    assert torch.equal(
        quantizer.candidate_ids[rows, local_ids],
        torch.tensor([9, 2]),
    )
    assert quantizer.quant_logits[0, 2] >= 3.25
    assert quantizer.quant_logits[1, 0] >= 4.25


def _group_scale_api():
    # Imported lazily so the pre-existing staged contracts still collect while
    # this new API is in its intentional red-test phase.
    from experiments.nowag_d6.staged_vector_gsq import (
        expand_group_scales,
        run_hidden_block_with_group_scales,
    )

    return expand_group_scales, run_hidden_block_with_group_scales


def test_group_scale_expansion_matches_exact_patched_block_forward() -> None:
    expand_group_scales, run_hidden_block_with_group_scales = _group_scale_api()
    torch.manual_seed(2)
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    hidden_states = torch.randn(2, 3, 5)
    weights = {
        name: module.weight.detach().clone() for name, module in modules.items()
    }
    group_size = 2
    group_scales = {}
    manually_scaled_weights = {}
    for index, (name, module) in enumerate(modules.items()):
        groups = (module.in_features + group_size - 1) // group_size
        scale = torch.linspace(
            0.75 + index * 0.01,
            1.25 + index * 0.01,
            steps=module.out_features * groups,
        ).reshape(module.out_features, groups)
        group_scales[name] = scale
        expanded = expand_group_scales(
            scale,
            in_features=module.in_features,
            group_size=group_size,
        )
        assert expanded.shape == module.weight.shape
        manually_scaled_weights[name] = weights[name] * expanded

    expected = run_hidden_stage_output(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        weights=manually_scaled_weights,
        objective="block",
    )
    actual = run_hidden_block_with_group_scales(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        weights=weights,
        group_scales=group_scales,
        group_size=group_size,
    )

    torch.testing.assert_close(actual, expected)


class _LogicalGroupScaleSource:
    def __init__(self) -> None:
        self.assignments = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.codebook = torch.tensor(
            [[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32
        )
        self.normalizer = _FakeNormalizer(
            col_norm=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            row_norm=torch.tensor([2.0, 3.0]),
        )


def test_logical_state_serializes_group_scales_and_counts_their_bits() -> None:
    from experiments.nowag_d6.run_prefix_vector_gsq import (
        logical_state,
        reconstruct_logical_state,
    )

    expand_group_scales, _ = _group_scale_api()
    source = _LogicalGroupScaleSource()
    legacy_state, legacy_bits = logical_state(
        source,
        source.assignments,
        (2, 4),
    )
    group_scale = torch.tensor([[2.0, 0.5], [0.25, 4.0]])

    state, logical_bits = logical_state(
        source,
        source.assignments,
        (2, 4),
        group_scale=group_scale,
        group_size=2,
    )

    assert torch.equal(state["group_scale"], group_scale.to(torch.float16))
    assert int(state["group_size"]) == 2
    assert logical_bits == legacy_bits + 16 * group_scale.numel()
    legacy_weight = reconstruct_logical_state(legacy_state, torch.device("cpu"))
    expected = legacy_weight * expand_group_scales(
        group_scale,
        in_features=4,
        group_size=2,
    )
    reconstructed = reconstruct_logical_state(state, torch.device("cpu"))
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)


def test_unit_group_scale_is_backward_compatible_with_legacy_checkpoint() -> None:
    from experiments.nowag_d6.run_prefix_vector_gsq import (
        logical_state,
        reconstruct_logical_state,
    )

    source = _LogicalGroupScaleSource()
    legacy_state, _ = logical_state(source, source.assignments, (2, 4))
    unit_state, _ = logical_state(
        source,
        source.assignments,
        (2, 4),
        group_scale=torch.ones(2, 2),
        group_size=2,
    )

    assert "group_scale" not in legacy_state
    assert "group_size" not in legacy_state
    legacy_weight = reconstruct_logical_state(legacy_state, torch.device("cpu"))
    unit_weight = reconstruct_logical_state(unit_state, torch.device("cpu"))
    torch.testing.assert_close(unit_weight, legacy_weight, rtol=0, atol=0)


def test_group_scale_expansion_truncates_the_final_partial_group() -> None:
    expand_group_scales, _ = _group_scale_api()
    group_scales = torch.tensor([[2.0, 3.0, 5.0]])

    expanded = expand_group_scales(
        group_scales,
        in_features=5,
        group_size=2,
    )

    torch.testing.assert_close(
        expanded,
        torch.tensor([[2.0, 2.0, 3.0, 3.0, 5.0]]),
        rtol=0,
        atol=0,
    )


def _apply_group_scales_api():
    from experiments.nowag_d6.staged_vector_gsq import apply_group_scales

    return apply_group_scales


def test_apply_group_scales_is_partial_and_multiplies_exact_expanded_groups() -> None:
    apply_group_scales = _apply_group_scales_api()
    modules = {
        "self_attn.q_proj": nn.Linear(5, 2, bias=False),
        "self_attn.k_proj": nn.Linear(5, 2, bias=False),
    }
    weights = {
        "self_attn.q_proj": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        ),
        "self_attn.k_proj": torch.tensor(
            [[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]]
        ),
    }
    originals = {name: weight.clone() for name, weight in weights.items()}
    q_scales = torch.tensor([[2.0, 3.0, 5.0], [0.5, 0.25, 4.0]])

    applied = apply_group_scales(
        weights,
        modules,
        {"self_attn.q_proj": q_scales},
        group_size=2,
    )

    expected_q = originals["self_attn.q_proj"] * torch.tensor(
        [[2.0, 2.0, 3.0, 3.0, 5.0], [0.5, 0.5, 0.25, 0.25, 4.0]]
    )
    torch.testing.assert_close(
        applied["self_attn.q_proj"], expected_q, rtol=0, atol=0
    )
    torch.testing.assert_close(
        applied["self_attn.k_proj"], originals["self_attn.k_proj"], rtol=0, atol=0
    )
    # The helper composes differentiable weights; it must not pre-fold scales
    # into the caller's fixed-assignment weight mapping.
    for name in weights:
        torch.testing.assert_close(weights[name], originals[name], rtol=0, atol=0)


def test_apply_group_scales_keeps_weight_and_log_scale_gradient_paths() -> None:
    apply_group_scales = _apply_group_scales_api()
    name = "mlp.down_proj"
    modules = {name: nn.Linear(5, 2, bias=False)}
    weight = nn.Parameter(torch.ones(2, 5))
    log_scale = nn.Parameter(torch.zeros(2, 3))

    applied = apply_group_scales(
        {name: weight},
        modules,
        {name: log_scale.exp()},
        group_size=2,
    )
    applied[name].sum().backward()

    assert weight.grad is not None
    assert log_scale.grad is not None
    assert torch.count_nonzero(weight.grad).item() == weight.numel()
    assert torch.count_nonzero(log_scale.grad).item() == log_scale.numel()


class _BlockLogicalSource:
    def __init__(self, rows: int, columns: int) -> None:
        # Values are deliberately not exactly representable in FP16.  This lets
        # the test distinguish deployable checkpoint reconstruction from a
        # shortcut that returns the FP32 training-time candidate.
        self.codebook = torch.tensor(
            [
                [0.33331, -0.14287],
                [0.27191, 0.61807],
                [-0.51713, 0.09179],
                [0.73117, -0.41119],
            ],
            dtype=torch.float32,
        )
        self.normalizer = _FakeNormalizer(
            col_norm=torch.linspace(0.9997, 1.0133, columns),
            row_norm=torch.linspace(1.0003, 1.0277, rows),
        )


def _block_logical_fixture():
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    sources = {}
    assignments = {}
    group_scales = {}
    group_size = 2
    for index, (name, module) in enumerate(modules.items()):
        source = _BlockLogicalSource(module.out_features, module.in_features)
        vector_groups = (module.in_features + 1) // 2
        ids = torch.arange(module.out_features * vector_groups).reshape(
            module.out_features, vector_groups
        )
        ids = (ids + index) % source.codebook.shape[0]
        sources[name] = source
        assignments[name] = ids
        scale_groups = (module.in_features + group_size - 1) // group_size
        group_scales[name] = torch.linspace(
            0.9973 + index * 0.0001,
            1.0167 + index * 0.0001,
            module.out_features * scale_groups,
        ).reshape(module.out_features, scale_groups)
    return modules, sources, assignments, group_scales, group_size


def test_block_logical_reconstruction_matches_per_linear_fp16_roundtrip() -> None:
    from experiments.nowag_d6.run_prefix_vector_gsq import (
        logical_state,
        reconstruct_logical_block_weights,
        reconstruct_logical_state,
    )

    expand_group_scales, _ = _group_scale_api()
    modules, sources, assignments, group_scales, group_size = (
        _block_logical_fixture()
    )
    device = torch.device("cpu")

    baseline = reconstruct_logical_block_weights(
        sources,
        assignments,
        modules,
        device,
        group_scales={},
        group_size=group_size,
    )
    candidate = reconstruct_logical_block_weights(
        sources,
        assignments,
        modules,
        device,
        group_scales=group_scales,
        group_size=group_size,
    )

    observed_rounding = False
    for name, module in modules.items():
        baseline_state, _ = logical_state(
            sources[name], assignments[name], tuple(module.weight.shape)
        )
        candidate_state, _ = logical_state(
            sources[name],
            assignments[name],
            tuple(module.weight.shape),
            group_scale=group_scales[name],
            group_size=group_size,
        )
        torch.testing.assert_close(
            baseline[name],
            reconstruct_logical_state(baseline_state, device),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate[name],
            reconstruct_logical_state(candidate_state, device),
            rtol=0,
            atol=0,
        )

        rows, columns = module.weight.shape
        vector_dim = sources[name].codebook.shape[1]
        padded_columns = (
            (columns + vector_dim - 1) // vector_dim
        ) * vector_dim
        fp32_training_weight = sources[name].codebook[
            assignments[name]
        ].reshape(rows, padded_columns)[:, :columns]
        fp32_training_weight = (
            fp32_training_weight
            * sources[name].normalizer.norms[1][:, None]
            * sources[name].normalizer.norms[0][None, :]
        )
        fp32_training_weight = fp32_training_weight * expand_group_scales(
            group_scales[name], columns, group_size
        )
        observed_rounding |= not torch.equal(
            candidate[name], fp32_training_weight
        )

        assert baseline_state["codebook"].dtype == torch.float16
        assert baseline_state["row_norm"].dtype == torch.float16
        assert baseline_state["col_norm"].dtype == torch.float16

    assert observed_rounding, "block reconstruction bypassed FP16 metadata rounding"


@pytest.mark.skip(reason="legacy shell launchers are not shipped; Python pipeline plan is tested")
def test_joint_group_scale_launcher_defaults_to_group_zero_zero(
    tmp_path: Path,
    ) -> None:
    project_root = Path(__file__).resolve().parents[1]
    launcher = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_tuning_local.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '100000\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    nowag_repo = tmp_path / "nowag"
    llama_model = tmp_path / "model"
    token_cache = tmp_path / "tokens.pt"
    nowag_repo.mkdir()
    llama_model.mkdir()
    token_cache.touch()
    env = os.environ.copy()
    for key in (
        "SCALE_FINETUNE_STEPS",
        "SCALE_FINETUNE_MODE",
        "BLOCK_POLISH_STEPS",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(tmp_path),
            "PYTHON": str(fake_python),
            "NOWAG_REPO": str(nowag_repo),
            "LLAMA_MODEL": str(llama_model),
            "TOKEN_CACHE": str(token_cache),
            "RESULT_ROOT": str(tmp_path / "results"),
            "RUN_ID": "joint-default-contract",
            "MODE": "gate",
            "STAGED_JOINT_GROUP_SCALES": "1",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = completed.stdout.splitlines()
    assert "--staged-joint-group-scales" in arguments
    assert arguments[arguments.index("--scale-finetune-mode") + 1] == "group"
    assert arguments[arguments.index("--scale-finetune-steps") + 1] == "0"
    assert arguments[arguments.index("--block-polish-steps") + 1] == "0"


@pytest.mark.skip(reason="legacy shell launchers are not shipped; Python pipeline plan is tested")
def test_rate_fair_group160_full_launcher_uses_one_gpu_and_full_protocol(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    launcher = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_jointgroup160_full_local.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CUDA=%s ARGS=%s\\n' \"$CUDA_VISIBLE_DEVICES\" \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '60000\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    nowag_repo = tmp_path / "nowag"
    llama_model = tmp_path / "model"
    token_cache = tmp_path / "tokens.pt"
    nowag_repo.mkdir()
    llama_model.mkdir()
    token_cache.touch()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(tmp_path),
            "PYTHON": str(fake_python),
            "NOWAG_REPO": str(nowag_repo),
            "LLAMA_MODEL": str(llama_model),
            "TOKEN_CACHE": str(token_cache),
            "RESULT_ROOT": str(tmp_path / "results"),
            "RUN_ID": "group160-full-contract",
            "CUDA_PHYSICAL_DEVICE": "6",
            "MIN_FREE_MIB": "45000",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [line for line in completed.stdout.splitlines() if line.startswith("CUDA=")]
    assert len(calls) == 3
    assert all(line.startswith("CUDA=6 ") for line in calls)
    quantization = calls[0].split("ARGS=", 1)[1].split()
    assert quantization[quantization.index("--max-layers") + 1] == "32"
    assert quantization[quantization.index("--gsq-objective") + 1] == "staged"
    assert "--staged-joint-group-scales" in quantization
    assert quantization[quantization.index("--scale-group-size") + 1] == "160"
    assert quantization[quantization.index("--scale-finetune-steps") + 1] == "0"
    assert quantization[quantization.index("--block-polish-steps") + 1] == "0"
    assert "evaluate_vector_checkpoint" in calls[1]
    assert "--ppl-seqlen 2048" in calls[1]
    assert "evaluate_logical_checkpoint_lm_eval" in calls[2]
    for task in (
        "arc_challenge",
        "arc_easy",
        "hellaswag",
        "lambada_openai",
        "piqa",
        "winogrande",
    ):
        assert task in calls[2]


class _JointToyQuantizer(nn.Module):
    def __init__(self, base_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_weight", base_weight.detach().clone())
        self.quant_logits = nn.Parameter(torch.zeros_like(base_weight))

    def forward(
        self,
        _temperature: float,
        _kappa: float,
        add_gumbel: bool = True,
    ):
        # The fixture is deterministic even when the production helper requests
        # Gumbel noise; its purpose is to expose every weight-gradient path.
        del add_gumbel
        weight = self.base_weight + self.quant_logits
        return weight, torch.empty(0), torch.empty(0)


def test_joint_refinement_loss_reaches_all_seven_logits_and_group_scales() -> None:
    from experiments.nowag_d6.staged_vector_gsq import joint_block_refinement_loss

    torch.manual_seed(3)
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    assert set(modules) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    quantizers = {
        name: _JointToyQuantizer(module.weight.detach())
        for name, module in modules.items()
    }
    group_size = 2
    log_group_scales = {
        name: nn.Parameter(
            torch.zeros(
                module.out_features,
                (module.in_features + group_size - 1) // group_size,
            )
        )
        for name, module in modules.items()
    }
    hidden_states = torch.randn(2, 3, 5)
    dense_outputs = torch.zeros_like(hidden_states)

    loss = joint_block_refinement_loss(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        dense_outputs=dense_outputs,
        quantizers=quantizers,
        log_group_scales=log_group_scales,
        temperature=1.0,
        kappa=1.0,
        group_size=group_size,
        add_gumbel=False,
    )
    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()

    for name in modules:
        logits_gradient = quantizers[name].quant_logits.grad
        scale_gradient = log_group_scales[name].grad
        assert logits_gradient is not None, f"missing quant-logit gradient: {name}"
        assert scale_gradient is not None, f"missing group-scale gradient: {name}"
        assert torch.isfinite(logits_gradient).all(), name
        assert torch.isfinite(scale_gradient).all(), name
        assert torch.count_nonzero(logits_gradient).item() > 0, name
        assert torch.count_nonzero(scale_gradient).item() > 0, name


def test_joint_best_snapshot_deep_copies_assignment_scale_pairs() -> None:
    from experiments.nowag_d6.staged_vector_gsq import (
        select_better_joint_snapshot,
        snapshot_joint_hard_state,
    )

    names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    initial_assignments = {
        name: torch.full((2, 2), index, dtype=torch.long)
        for index, name in enumerate(names)
    }
    initial_logs = {
        name: torch.full((2, 2), index / 100.0)
        for index, name in enumerate(names)
    }
    best = snapshot_joint_hard_state(3.0, initial_assignments, initial_logs)

    step_assignments = {
        name: value + 10 for name, value in initial_assignments.items()
    }
    step_logs = {name: value + 0.125 for name, value in initial_logs.items()}
    best = select_better_joint_snapshot(
        best,
        2.0,
        step_assignments,
        step_logs,
    )
    for name in names:
        assert torch.equal(best.assignments[name], step_assignments[name])
        assert torch.equal(best.log_group_scales[name], step_logs[name])
        assert best.assignments[name].data_ptr() != step_assignments[name].data_ptr()
        assert best.log_group_scales[name].data_ptr() != step_logs[name].data_ptr()

    worse_assignments = {
        name: value + 100 for name, value in initial_assignments.items()
    }
    worse_logs = {name: value - 0.5 for name, value in initial_logs.items()}
    unchanged = select_better_joint_snapshot(
        best,
        2.5,
        worse_assignments,
        worse_logs,
    )
    assert unchanged.loss == 2.0
    for name in names:
        assert torch.equal(unchanged.assignments[name], step_assignments[name])
        assert torch.equal(unchanged.log_group_scales[name], step_logs[name])

    # A saved best state must remain immutable after the live training tensors
    # advance to the next optimizer step.
    for tensor in step_assignments.values():
        tensor.add_(1000)
    for tensor in step_logs.values():
        tensor.add_(1000)
    for index, name in enumerate(names):
        assert torch.equal(
            unchanged.assignments[name],
            torch.full((2, 2), index + 10, dtype=torch.long),
        )
        torch.testing.assert_close(
            unchanged.log_group_scales[name],
            torch.full((2, 2), index / 100.0 + 0.125),
            rtol=0,
            atol=1e-7,
        )


def test_joint_group_scales_can_continue_into_block_refinement(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    runner_source = (
        project_root / "experiments/nowag_d6/run_prefix_vector_gsq.py"
    ).read_text(encoding="utf-8")
    assert (
        "block polish does not yet preserve staged joint group scales"
        not in runner_source
    )


@pytest.mark.skip(reason="legacy shell launchers are not shipped; Python pipeline plan is tested")
def test_budget256_group160_gate_launcher_uses_configurable_disjoint_budgets(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    core = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_tuning_local.sh"
    ).read_text(encoding="utf-8")
    gate = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_jointgroup160_budget256_gate_local.sh"
    ).read_text(encoding="utf-8")
    full = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_jointgroup160_budget256_full_local.sh"
    ).read_text(encoding="utf-8")
    assert 'CALIB_SEQUENCES="${CALIB_SEQUENCES:-64}"' in core
    assert 'TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-8}"' in core
    assert 'VALIDATION_SEQUENCES="${VALIDATION_SEQUENCES:-8}"' in core
    assert '--calib-sequences "${CALIB_SEQUENCES}"' in core
    assert '--train-sequences "${TRAIN_SEQUENCES}"' in core
    assert '--validation-sequences "${VALIDATION_SEQUENCES}"' in core
    assert "export CALIB_SEQUENCES=256" in gate
    assert "export TRAIN_SEQUENCES=32" in gate
    assert "export VALIDATION_SEQUENCES=32" in gate
    assert "export MODE=gate" in gate
    assert "export SCALE_GROUP_SIZE=160" in gate
    assert 'exec bash "${SCRIPT_DIR}/run_llama3_8b_staged_tuning_local.sh"' in gate
    assert "export CALIB_SEQUENCES=256" in full
    assert "export TRAIN_SEQUENCES=32" in full
    assert "export VALIDATION_SEQUENCES=32" in full
    assert "export MODE=full" in full
    assert "export SCALE_GROUP_SIZE=160" in full
    assert 'exec bash "${SCRIPT_DIR}/run_llama3_8b_staged_tuning_local.sh"' in full

    launcher = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_tuning_local.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '100000\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    nowag_repo = tmp_path / "nowag"
    llama_model = tmp_path / "model"
    token_cache = tmp_path / "tokens.pt"
    nowag_repo.mkdir()
    llama_model.mkdir()
    token_cache.touch()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_ROOT": str(tmp_path),
            "PYTHON": str(fake_python),
            "NOWAG_REPO": str(nowag_repo),
            "LLAMA_MODEL": str(llama_model),
            "TOKEN_CACHE": str(token_cache),
            "RESULT_ROOT": str(tmp_path / "results"),
            "RUN_ID": "joint-refinement-contract",
            "MODE": "gate",
            "STAGED_JOINT_GROUP_SCALES": "1",
            "SCALE_FINETUNE_MODE": "group",
            "SCALE_FINETUNE_STEPS": "0",
            "BLOCK_POLISH_STEPS": "7",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = completed.stdout.splitlines()
    assert "--staged-joint-group-scales" in arguments
    assert arguments[arguments.index("--block-polish-steps") + 1] == "7"


@pytest.mark.skip(reason="legacy shell launchers are not shipped; Python pipeline plan is tested")
def test_official_gsq_calibration_launcher_matches_paper_data_protocol() -> None:
    project_root = Path(__file__).resolve().parents[1]
    core = (
        project_root
        / "experiments/llama_qtip/run_llama3_8b_staged_tuning_local.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        project_root
        / "experiments/llama_qtip/"
        "run_llama3_8b_staged_jointgroup160_official_gsq_gate_local.sh"
    ).read_text(encoding="utf-8")

    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"' in core
    assert 'TRAIN_MICROBATCH_SIZE="${TRAIN_MICROBATCH_SIZE:-1}"' in core
    assert '--train-batch-size "${TRAIN_BATCH_SIZE}"' in core
    assert '--train-microbatch-size "${TRAIN_MICROBATCH_SIZE}"' in core

    expected_exports = {
        "MODE": "gate",
        "CALIB_SEQUENCES": "512",
        "TRAIN_SEQUENCES": "4096",
        "VALIDATION_SEQUENCES": "128",
        "SEGMENT_LENGTH": "0",
        "TRAIN_BATCH_SIZE": "64",
        "TRAIN_MICROBATCH_SIZE": "2",
        "STAGE_STEPS_QK": "640",
        "STAGE_STEPS_VO": "640",
        "STAGE_STEPS_MLP": "640",
    }
    for name, value in expected_exports.items():
        assert f"export {name}={value}" in launcher
    assert "officialgsq4096_128_512" in launcher
    assert "remote14" not in launcher
    assert 'exec bash "${SCRIPT_DIR}/run_llama3_8b_staged_tuning_local.sh"' in launcher


def test_dense_stage_targets_never_use_hybrid_quantized_references() -> None:
    from experiments.nowag_d6.staged_vector_gsq import collect_dense_stage_targets

    torch.manual_seed(4)
    model = _TinyLlama()
    modules = _tiny_linear_modules(model)
    hidden_states = torch.randn(2, 3, 5)
    selected_weights = {
        name: torch.zeros_like(module.weight) for name, module in modules.items()
    }
    cached_dense_block = (
        run_hidden_stage_output(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            weights=None,
            objective="block",
        ).detach().to("cpu", torch.bfloat16)
        + torch.tensor(0.125, dtype=torch.bfloat16)
    )

    first_targets = {}
    for objective in ("linear_q", "linear_k", "attention"):
        dense_teacher = run_hidden_stage_output(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            weights=None,
            objective=objective,
        ).detach().to("cpu", torch.bfloat16)
        hybrid_target = run_hidden_stage_output(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            weights=selected_weights,
            objective=objective,
        ).detach().to("cpu", torch.bfloat16)
        target = collect_dense_stage_targets(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            dense_block_outputs=cached_dense_block,
            objective=objective,
            batch_size=1,
        )
        torch.testing.assert_close(target, dense_teacher, rtol=0, atol=0)
        assert not torch.equal(target, hybrid_target), objective
        first_targets[objective] = target.clone()

    block_target = collect_dense_stage_targets(
        model,
        layer_index=0,
        modules=modules,
        hidden_states=hidden_states,
        dense_block_outputs=cached_dense_block,
        objective="block",
        batch_size=1,
    )
    assert torch.equal(block_target, cached_dense_block)

    # Changing the candidate/selected quantized prefix cannot alter the dense
    # teacher.  It is deliberately not an input to the target collector.
    for name in selected_weights:
        selected_weights[name].normal_(mean=3.0, std=2.0)
    for objective, first in first_targets.items():
        repeated = collect_dense_stage_targets(
            model,
            layer_index=0,
            modules=modules,
            hidden_states=hidden_states,
            dense_block_outputs=cached_dense_block,
            objective=objective,
            batch_size=2,
        )
        torch.testing.assert_close(repeated, first, rtol=0, atol=0)
