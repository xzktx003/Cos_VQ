import json
from types import SimpleNamespace

import pytest
import torch

from experiments.llama_qtip.build_matched_llama_assignment_cache import (
    combine_fineweb_and_guard_tokens,
    load_c4_validation,
    verify_cache_token_domain,
    verify_prefix_extension,
    verify_raw_tokenizer_equivalence,
)


def test_public_c4_loader_uses_complete_english_validation_split(monkeypatch):
    calls = []
    dataset = object()

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(
        "experiments.llama_qtip.build_matched_llama_assignment_cache.load_dataset",
        fake_load_dataset,
    )
    assert load_c4_validation(source="download", revision="dataset-revision") is dataset
    assert calls == [
        (("allenai/c4", "en"), {"split": "validation", "revision": "dataset-revision"})
    ]


class _Tokenizer:
    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        bos_token_id: int = 10,
        offset: int = 0,
    ) -> None:
        self._vocab = vocab or {"a": 2, "b": 3}
        self.bos_token_id = bos_token_id
        self.offset = offset
        regular_vocab = {
            token: token_id
            for token, token_id in self._vocab.items()
            if token_id < bos_token_id
        }
        added_tokens = [
            {"id": token_id, "content": token}
            for token, token_id in self._vocab.items()
            if token_id >= bos_token_id
        ]
        self.backend_tokenizer = SimpleNamespace(
            to_str=lambda: json.dumps(
                {
                    "model": {"vocab": regular_vocab, "offset": offset},
                    "added_tokens": added_tokens,
                },
                sort_keys=True,
            )
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = [self._vocab.get(char, 0) + self.offset for char in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values


def test_tokenizer_equivalence_accepts_identical_raw_encoding() -> None:
    verify_raw_tokenizer_equivalence(_Tokenizer(), _Tokenizer())


def test_tokenizer_equivalence_allows_unused_reserved_special_token_renames() -> None:
    reference = _Tokenizer(vocab={"a": 2, "b": 3, "<reserved>": 11})
    target = _Tokenizer(vocab={"a": 2, "b": 3, "<new-special>": 11})
    verify_raw_tokenizer_equivalence(reference, target)


@pytest.mark.parametrize(
    "target",
    [
        _Tokenizer(vocab={"a": 2, "b": 4}),
        _Tokenizer(bos_token_id=9),
        _Tokenizer(offset=1),
    ],
    ids=["vocab", "bos", "backend"],
)
def test_tokenizer_equivalence_rejects_nonidentical_raw_encoding(
    target: _Tokenizer,
) -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        verify_raw_tokenizer_equivalence(_Tokenizer(), target)


def test_combined_cache_preserves_fineweb_prefix_and_records_guard_provenance() -> None:
    fineweb = {
        "identity": "sample-10BT|reference|len=4|n=3",
        "input_ids": torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=torch.int64,
        ),
    }
    guard = [
        torch.tensor([13, 14, 15, 16]),
        torch.tensor([17, 18, 19, 20]),
    ]
    payload = combine_fineweb_and_guard_tokens(
        fineweb,
        guard,
        target_model="/models/llama3.1",
        guard_dataset="local-c4-validation",
    )
    assert tuple(payload["input_ids"].shape) == (5, 4)
    assert torch.equal(payload["input_ids"][:3], fineweb["input_ids"])
    assert payload["fineweb_rows"] == [0, 2]
    assert payload["guard_rows"] == [3, 4]
    assert payload["target_model"] == "/models/llama3.1"
    assert payload["guard_dataset"] == "local-c4-validation"


def test_combined_cache_rejects_wrong_guard_shape_or_dtype() -> None:
    fineweb = {
        "identity": "fineweb",
        "input_ids": torch.ones(2, 4, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="length"):
        combine_fineweb_and_guard_tokens(
            fineweb,
            [torch.ones(3, dtype=torch.int64)],
            target_model="model",
            guard_dataset="guard",
        )
    with pytest.raises(ValueError, match="integer"):
        combine_fineweb_and_guard_tokens(
            fineweb,
            [torch.ones(4, dtype=torch.float32)],
            target_model="model",
            guard_dataset="guard",
        )


def test_prefix_extension_requires_bit_exact_existing_rows() -> None:
    prefix = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
    }
    extended = {
        "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64),
    }
    assert verify_prefix_extension(prefix, extended) == 2
    with pytest.raises(ValueError, match="bit-exact"):
        verify_prefix_extension(
            prefix,
            {"input_ids": torch.tensor([[1, 2], [3, 9], [5, 6]])},
        )
    with pytest.raises(ValueError, match="strictly extend"):
        verify_prefix_extension(prefix, prefix)


def test_cache_domain_accepts_regular_tokens_and_shared_bos_only() -> None:
    verify_cache_token_domain(
        torch.tensor([[0, 2, 9, 10]], dtype=torch.int64),
        regular_vocab_size=10,
        bos_token_id=10,
    )


def test_cache_domain_rejects_other_special_tokens() -> None:
    with pytest.raises(ValueError, match="unsupported special token"):
        verify_cache_token_domain(
            torch.tensor([[0, 10, 11]], dtype=torch.int64),
            regular_vocab_size=10,
            bos_token_id=10,
        )
