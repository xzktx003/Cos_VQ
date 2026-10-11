from pathlib import Path
import inspect

import torch
import torch.nn.functional as F
import pytest

import experiments.nowag_d6.probe_text_constrained_assignment_gain as text_probe
from experiments.nowag_d6.probe_text_constrained_assignment_gain import (
    CachedTextBatch,
    PrefixTextCache,
    build_deferred_text_teacher_split,
    choose_text_constrained_bundle,
    chunked_topk_teacher_cross_entropy,
    estimate_prefix_text_cache_bytes,
    score_cached_text_candidates,
    select_text_cache_parity_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lexicographic_acceptor_skips_task_best_when_text_is_infeasible() -> None:
    decision = choose_text_constrained_bundle(
        incumbent_task_loss=1.0,
        incumbent_text_loss=2.0,
        candidates={
            "task_best": {"task_loss": 0.80, "text_loss": 2.01},
            "text_safe": {"task_loss": 0.84, "text_loss": 1.99},
            "task_worse": {"task_loss": 1.01, "text_loss": 1.90},
        },
        task_tolerance=0.0,
        text_tolerance=0.0,
    )

    assert decision["accepted"] is True
    assert decision["key"] == "text_safe"
    assert decision["task_gain"] == pytest.approx(0.16)
    assert decision["text_change"] == pytest.approx(-0.01)
    assert decision["feasible_keys"] == ["text_safe"]


def test_lexicographic_acceptor_rejects_when_text_feasible_set_is_empty() -> None:
    decision = choose_text_constrained_bundle(
        incumbent_task_loss=1.0,
        incumbent_text_loss=2.0,
        candidates={
            "text_bad": {"task_loss": 0.90, "text_loss": 2.001},
            "task_bad": {"task_loss": 1.01, "text_loss": 1.90},
        },
        task_tolerance=0.0,
        text_tolerance=0.0,
    )

    assert decision["accepted"] is False
    assert decision["key"] is None
    assert decision["feasible_keys"] == []
    assert decision["rejections"] == {
        "task_bad": "task_non_improving",
        "text_bad": "text_constraint",
    }


def test_chunked_teacher_cross_entropy_matches_full_vocabulary() -> None:
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(3, 5, 7, generator=generator)
    weight = torch.randn(13, 7, generator=generator)
    teacher_indices = torch.randint(0, 13, (3, 5, 4), generator=generator)
    teacher_probabilities = torch.rand(3, 5, 4, generator=generator)
    teacher_probabilities /= teacher_probabilities.sum(dim=-1, keepdim=True)

    logits = F.linear(hidden, weight)
    full_log_probabilities = logits.log_softmax(dim=-1)
    expected = -(
        teacher_probabilities
        * full_log_probabilities.gather(2, teacher_indices)
    ).sum(dim=-1).mean(dim=-1)

    actual = chunked_topk_teacher_cross_entropy(
        hidden,
        weight,
        teacher_indices,
        teacher_probabilities,
        vocab_chunk_size=5,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_prefix_text_cache_size_contract_is_exact() -> None:
    assert estimate_prefix_text_cache_bytes(
        sequences=4096,
        sequence_length=4096,
        hidden_size=4096,
        dtype_bytes=2,
    ) == 137_438_953_472


def test_candidate_batched_text_scores_match_independent_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInner:
        def __init__(self) -> None:
            self.layers = [object(), object()]
            self.norm = torch.nn.Identity()

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeInner()
            self.lm_head = torch.nn.Linear(2, 5, bias=False)

    model = FakeModel()
    with torch.no_grad():
        model.lm_head.weight.copy_(
            torch.tensor(
                [
                    [0.2, -0.1],
                    [0.3, 0.4],
                    [-0.5, 0.1],
                    [0.7, -0.2],
                    [-0.4, -0.3],
                ]
            )
        )

    def fake_run_hidden(
        _model: object,
        _layer_index: int,
        _modules: object,
        hidden_states: torch.Tensor,
        weights: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not weights:
            return hidden_states
        return hidden_states + next(iter(weights.values())).reshape(1, 1, 1)

    monkeypatch.setattr(text_probe, "get_linear_modules", lambda *_args: {})
    monkeypatch.setattr(text_probe, "run_hidden_block_output", fake_run_hidden)

    hidden = torch.tensor(
        [[[0.1, 0.2], [0.3, -0.1], [0.0, 0.4], [-0.2, 0.5]]]
    )
    teacher_indices = torch.tensor([[[0, 1], [2, 3], [1, 4]]])
    teacher_probabilities = torch.tensor(
        [[[0.6, 0.4], [0.25, 0.75], [0.8, 0.2]]]
    )
    cache = PrefixTextCache(
        start_layer=0,
        batches=[
            CachedTextBatch(
                hidden_states=hidden,
                input_ids=torch.ones(1, 4, dtype=torch.long),
                teacher_indices=teacher_indices,
                teacher_probabilities=teacher_probabilities,
                rows=(0,),
            )
        ],
        example_count=1,
        token_count=4,
        hidden_bytes=hidden.numel() * hidden.element_size(),
    )
    overrides = {
        (0, "a"): {"a": torch.tensor(0.1)},
        (0, "b"): {"b": torch.tensor(-0.2)},
    }

    actual = score_cached_text_candidates(
        model,
        cache,
        layer=0,
        candidate_overrides=overrides,
        vocab_chunk_size=2,
        device=torch.device("cpu"),
    )
    expected_a = chunked_topk_teacher_cross_entropy(
        (hidden + 0.1)[:, :-1],
        model.lm_head.weight,
        teacher_indices,
        teacher_probabilities,
        vocab_chunk_size=2,
    )[0]
    expected_b = chunked_topk_teacher_cross_entropy(
        (hidden - 0.2)[:, :-1],
        model.lm_head.weight,
        teacher_indices,
        teacher_probabilities,
        vocab_chunk_size=2,
    )[0]

    assert actual[(0, "a")] == pytest.approx(float(expected_a))
    assert actual[(0, "b")] == pytest.approx(float(expected_b))


def test_text_cache_parity_indices_span_the_full_train_rows() -> None:
    indices = select_text_cache_parity_indices(example_count=4096, count=8)

    assert indices[0] == 0
    assert indices[-1] == 4095
    assert len(indices) == len(set(indices)) == 8


def test_deferred_teacher_keeps_only_one_gpu_model_resident() -> None:
    source = inspect.getsource(build_deferred_text_teacher_split)

    offload = source.index('model.to("cpu")')
    dense_load = source.index("AutoModelForCausalLM.from_pretrained")
    dense_delete = source.index("del dense_teacher")
    quantized_reload = source.index("model.to(device)")
    assert offload < dense_load < dense_delete < quantized_reload

