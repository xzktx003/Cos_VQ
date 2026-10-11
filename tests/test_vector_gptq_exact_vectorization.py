import sys
import types

import pytest
import torch
import torch.nn as nn


if "lion_pytorch" not in sys.modules:
    sys.modules["lion_pytorch"] = types.SimpleNamespace(Lion=object)

from experiments.nowag_d6 import diagnose_vector_gsq_gradients as vector_gsq


def legacy_candidate_losses(
    base: torch.Tensor,
    candidates: torch.Tensor,
    upper_block: torch.Tensor,
) -> torch.Tensor:
    """Reference the original six-step GPTQ candidate simulation."""
    simulated = base.unsqueeze(1).expand(-1, candidates.shape[0], -1).clone()
    losses = torch.zeros(simulated.shape[:2], dtype=base.dtype)
    for offset in range(base.shape[1]):
        difference = simulated[:, :, offset] - candidates[:, offset].unsqueeze(0)
        diagonal = upper_block[offset, offset].clamp_min(1e-12)
        losses.add_(difference.square() / diagonal.square())
        error = difference / diagonal
        simulated[:, :, offset:] -= (
            error.unsqueeze(2) * upper_block[offset, offset:].view(1, 1, -1)
        )
    return losses


@pytest.mark.parametrize("width", [1, 3, 6])
def test_vectorized_candidate_losses_match_legacy_sequential_feedback(width: int) -> None:
    generator = torch.Generator().manual_seed(100 + width)
    base = torch.randn(7, width, generator=generator)
    candidates = torch.randn(31, width, generator=generator)
    upper_block = torch.triu(torch.randn(width, width, generator=generator) * 0.2)
    upper_block.diagonal().add_(1.5)

    expected = legacy_candidate_losses(base, candidates, upper_block)
    actual = vector_gsq._vector_gptq_candidate_losses(
        base, candidates, upper_block
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.equal(actual.argmin(dim=1), expected.argmin(dim=1))


def test_vectorized_candidate_losses_preserve_legacy_diagonal_floor() -> None:
    base = torch.tensor([[1e-12, -2e-12]])
    candidates = torch.tensor([[0.0, 0.0], [1e-12, -1e-12]])
    upper_block = torch.tensor([[1e-14, 2e-13], [0.0, 8e-13]])

    expected = legacy_candidate_losses(base, candidates, upper_block)
    actual = vector_gsq._vector_gptq_candidate_losses(
        base, candidates, upper_block
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.equal(actual.argmin(dim=1), expected.argmin(dim=1))


def test_top128_codebook_neighborhood_contains_its_anchor_once() -> None:
    generator = torch.Generator().manual_seed(128)
    codebook = torch.randn(256, 6, generator=generator)

    neighbors = vector_gsq._build_codebook_neighbor_table(
        codebook, candidate_count=128, row_chunk=37
    )

    assert neighbors.shape == (256, 128)
    assert torch.equal(neighbors[:, 0], torch.arange(256))
    assert all(torch.unique(row).numel() == 128 for row in neighbors)


def test_top128_codebook_neighborhood_caps_at_codebook_size() -> None:
    codebook = torch.randn(17, 6, generator=torch.Generator().manual_seed(17))

    neighbors = vector_gsq._build_codebook_neighbor_table(
        codebook, candidate_count=128
    )

    assert neighbors.shape == (17, 17)
    assert torch.equal(neighbors[:, 0], torch.arange(17))


def test_per_row_top128_distance_kernel_matches_direct_distance() -> None:
    generator = torch.Generator().manual_seed(812)
    values = torch.randn(4, 6, generator=generator)
    candidates = torch.randn(4, 128, 6, generator=generator)

    actual = vector_gsq._projected_squared_distances(values, candidates)
    expected = (values.unsqueeze(1) - candidates).square().sum(dim=2)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


class IdentityNormalizer:
    def __init__(self, columns: int) -> None:
        self.norms = [torch.ones(columns)]

    def normalize(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def denormalize(self, weight: torch.Tensor) -> torch.Tensor:
        return weight


class DummyVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        rows: int,
        columns: int,
    ) -> None:
        super().__init__()
        self.codebook = codebook
        self.assignments = assignments.clone()
        self.rows = rows
        self.columns = columns
        self.normalizer = IdentityNormalizer(columns)

    def reconstruct(self) -> torch.Tensor:
        return self.codebook[self.assignments].reshape(self.rows, self.columns)


def legacy_vector_gptq_refinement(
    original: torch.Tensor,
    codebook: torch.Tensor,
    hessian: torch.Tensor,
    vector_dim: int,
    percdamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = original.shape[1]
    transformed = hessian.clone()
    diagonal_indices = torch.arange(columns)
    transformed[diagonal_indices, diagonal_indices] += (
        percdamp * torch.diag(transformed).mean()
    )
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(transformed))
    upper = torch.linalg.cholesky(inverse, upper=True)
    work = original.clone()
    quantized = torch.zeros_like(work)
    assignment_groups = []

    for start in range(0, columns, vector_dim):
        end = min(start + vector_dim, columns)
        upper_block = upper[start:end, start:end]
        candidates = codebook[:, : end - start]
        selected_ids = legacy_candidate_losses(
            work[:, start:end], candidates, upper_block
        ).argmin(dim=1)
        selected = candidates[selected_ids]
        errors = torch.zeros_like(selected)
        for offset in range(end - start):
            diagonal = upper_block[offset, offset].clamp_min(1e-12)
            difference = work[:, start + offset] - selected[:, offset]
            error = difference / diagonal
            work[:, start + offset : end] -= (
                error.unsqueeze(1) * upper_block[offset, offset:].unsqueeze(0)
            )
            errors[:, offset] = error
        quantized[:, start:end] = selected
        if end < columns:
            work[:, end:] -= errors @ upper[start:end, end:]
        assignment_groups.append(selected_ids)

    return quantized, torch.stack(assignment_groups, dim=1).reshape(-1)


def test_vectorized_vector_gptq_matches_legacy_weights_and_assignments() -> None:
    generator = torch.Generator().manual_seed(20260812)
    rows, columns, vector_dim = 4, 6, 3
    codebook = torch.randn(8, vector_dim, generator=generator)
    target_assignments = torch.tensor([1, 6, 3, 4, 7, 2, 5, 1])
    original = codebook[target_assignments].reshape(rows, columns)
    original = original + 0.01 * torch.randn(
        rows, columns, generator=generator
    )
    initial_assignments = torch.zeros(rows * (columns // vector_dim), dtype=torch.long)
    source = DummyVectorQuantizer(
        codebook, initial_assignments, rows=rows, columns=columns
    )
    activations = torch.randn(32, columns, generator=generator)
    hessian = activations.T @ activations / activations.shape[0]

    expected_weight, expected_assignments = legacy_vector_gptq_refinement(
        original, codebook, hessian, vector_dim=vector_dim, percdamp=0.1
    )
    actual_weight, actual_assignments, stats = (
        vector_gsq.vector_codebook_aware_gptq(
            source,
            original,
            hessian,
            vector_dim=vector_dim,
            percdamp=0.1,
            row_chunk=2,
            candidate_count=codebook.shape[0],
        )
    )

    assert stats["gptq_refinement_accepted"] == 1.0
    assert stats["gptq_candidate_count"] == codebook.shape[0]
    assert torch.equal(actual_assignments, expected_assignments)
    assert torch.equal(source.assignments, expected_assignments)
    torch.testing.assert_close(actual_weight, expected_weight, rtol=1e-6, atol=1e-6)


def test_vector_gptq_defaults_to_top128_around_nowag_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(4096)
    rows, columns, vector_dim = 3, 6, 3
    codebook = torch.randn(256, vector_dim, generator=generator)
    initial_assignments = torch.tensor([0, 17, 31, 63, 127, 255])
    source = DummyVectorQuantizer(
        codebook, initial_assignments, rows=rows, columns=columns
    )
    original = torch.randn(rows, columns, generator=generator)
    activations = torch.randn(24, columns, generator=generator)
    hessian = activations.T @ activations / activations.shape[0]
    objectives = iter([2.0, 1.0])
    monkeypatch.setattr(
        vector_gsq,
        "full_hessian_objective",
        lambda *_args: next(objectives),
    )

    _, assignments, stats = vector_gsq.vector_codebook_aware_gptq(
        source,
        original,
        hessian,
        vector_dim=vector_dim,
        row_chunk=2,
    )

    neighbors = vector_gsq._build_codebook_neighbor_table(
        codebook, candidate_count=128, row_chunk=2
    )
    initial_matrix = initial_assignments.reshape(rows, columns // vector_dim)
    selected_matrix = assignments.reshape_as(initial_matrix)
    assert stats["gptq_candidate_count"] == 128
    assert stats["gptq_codebook_size"] == 256
    assert stats["gptq_candidate_fraction"] == 0.5
    for row in range(rows):
        for group in range(columns // vector_dim):
            assert selected_matrix[row, group] in neighbors[initial_matrix[row, group]]


def test_vector_gptq_rejects_a_worse_refinement_without_mutating_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codebook = torch.tensor(
        [[-1.0, -0.5], [0.0, 0.25], [0.75, 1.0]], dtype=torch.float32
    )
    initial_assignments = torch.tensor([0, 2, 1, 1])
    source = DummyVectorQuantizer(codebook, initial_assignments, rows=2, columns=4)
    original = torch.tensor(
        [[-0.8, -0.4, 0.6, 0.9], [0.1, 0.2, -0.1, 0.3]], dtype=torch.float32
    )
    hessian = torch.tensor(
        [
            [1.2, 0.1, 0.0, 0.0],
            [0.1, 1.1, 0.1, 0.0],
            [0.0, 0.1, 1.3, 0.2],
            [0.0, 0.0, 0.2, 1.4],
        ],
        dtype=torch.float32,
    )
    objectives = iter([1.0, 2.0])
    monkeypatch.setattr(
        vector_gsq,
        "full_hessian_objective",
        lambda *_args: next(objectives),
    )

    weight, assignments, stats = vector_gsq.vector_codebook_aware_gptq(
        source, original, hessian, vector_dim=2, row_chunk=1
    )

    torch.testing.assert_close(weight, source.reconstruct())
    assert torch.equal(assignments, initial_assignments)
    assert torch.equal(source.assignments, initial_assignments)
    assert stats["gptq_refinement_accepted"] == 0.0
    assert stats["assignment_change_rate"] == 0.0
