import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.nowag_d6.logical_state import (
    LINEAR_NAMES,
    get_linear_modules,
    reconstruct_logical_state,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_module_import_does_not_import_transformers() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from experiments.nowag_d6.logical_state import LINEAR_NAMES, get_linear_modules, reconstruct_logical_state
assert reconstruct_logical_state.__module__ == "experiments.nowag_d6.logical_state"
assert get_linear_modules.__module__ == "experiments.nowag_d6.logical_state"
assert len(LINEAR_NAMES) == 7
assert "transformers" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_get_linear_modules_returns_all_seven_toy_block_linears() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(3, 2)
            self.k_proj = torch.nn.Linear(3, 2)
            self.v_proj = torch.nn.Linear(3, 2)
            self.o_proj = torch.nn.Linear(2, 3)

    class Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(3, 4)
            self.up_proj = torch.nn.Linear(3, 4)
            self.down_proj = torch.nn.Linear(4, 3)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = Mlp()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block()])

    modules = get_linear_modules(Model(), 0)

    assert tuple(modules) == LINEAR_NAMES
    assert all(isinstance(module, torch.nn.Linear) for module in modules.values())


def test_reconstruct_logical_state_handles_padding_and_group_scale() -> None:
    state = {
        "shape": torch.tensor([2, 5], dtype=torch.int32),
        "vector_dim": torch.tensor(2, dtype=torch.int32),
        "assignments": torch.tensor([0, 1, 2, 2, 3, 1], dtype=torch.int32),
        "codebook": torch.tensor(
            [[1.0, -1.0], [0.5, 2.0], [-3.0, 4.0], [1.5, -0.25]],
            dtype=torch.float16,
        ),
        "row_norm": torch.tensor([2.0, 0.5], dtype=torch.float16),
        "col_norm": torch.tensor([1.0, 2.0, 0.5, 4.0, 3.0], dtype=torch.float16),
        "group_size": torch.tensor(3, dtype=torch.int32),
        "group_scale": torch.tensor([[1.0, 0.25], [2.0, 0.5]], dtype=torch.float16),
    }

    weight = reconstruct_logical_state(state, torch.device("cpu"))

    expected_normalized = torch.tensor(
        [
            [1.0, -1.0, 0.5, 2.0, -3.0],
            [-3.0, 4.0, 1.5, -0.25, 0.5],
        ],
        dtype=torch.float32,
    )
    expected_group_scale = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.25, 0.25],
            [2.0, 2.0, 2.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    expected = (
        expected_normalized
        * state["row_norm"].float()[:, None]
        * state["col_norm"].float()[None, :]
        * expected_group_scale
    )
    torch.testing.assert_close(weight, expected)


def test_group_scale_validation_matches_expand_group_scales_contract() -> None:
    state = {
        "shape": torch.tensor([1, 4], dtype=torch.int32),
        "vector_dim": torch.tensor(2, dtype=torch.int32),
        "assignments": torch.tensor([0, 1], dtype=torch.int32),
        "codebook": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
        "row_norm": torch.ones(1, dtype=torch.float16),
        "col_norm": torch.ones(4, dtype=torch.float16),
        "group_scale": torch.ones(1, 2, dtype=torch.float16),
    }
    with pytest.raises(ValueError, match="missing group_size"):
        reconstruct_logical_state(state, torch.device("cpu"))

    state["group_size"] = torch.tensor(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="group_size must be positive"):
        reconstruct_logical_state(state, torch.device("cpu"))

    state["group_size"] = torch.tensor(3, dtype=torch.int32)
    state["group_scale"] = torch.ones(1, 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="expected 2"):
        reconstruct_logical_state(state, torch.device("cpu"))
