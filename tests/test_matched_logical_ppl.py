import math
import types

import pytest
import torch

from experiments.llama_qtip import evaluate_qtip_checkpoint as qtip_eval
from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as logical_eval


class _Encoded:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


class _ToyTokenizer:
    pad_token_id = 0

    def __call__(self, _text: str, *, return_tensors: str) -> _Encoded:
        assert return_tensors == "pt"
        return _Encoded(torch.tensor([[0, 1, 0, 2, 2, 0, 3, 1]]))


class _NextTokenToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 1)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        for position in range(input_ids.shape[1] - 1):
            targets = input_ids[:, position + 1]
            logits[:, position, :].scatter_(1, targets[:, None], 1.25)
        return types.SimpleNamespace(logits=logits)


def test_qtip_wikitext2_ppl_uses_shifted_all_token_ce_including_pad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"text": ["ignored"]}

    monkeypatch.setattr(
        qtip_eval,
        "load_dataset",
        fake_load_dataset,
    )

    ppl, sequence_count = qtip_eval.evaluate_wikitext2(
        _NextTokenToyModel(),
        _ToyTokenizer(),
        seqlen=4,
        batch_size=1,
        revision="wikitext-commit",
    )

    assert calls == [
        (
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            {"split": "test", "revision": "wikitext-commit"},
        )
    ]
    per_token_loss = math.log(math.exp(1.25) + 3.0) - 1.25
    assert sequence_count == 2
    assert ppl == pytest.approx(math.exp(per_token_loss))


def test_write_matched_ppl_calls_qtip_metric_and_persists_json(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_evaluate_wikitext2(
        model, tokenizer, *, seqlen: int, batch_size: int, revision: str
    ):
        calls.append((model, tokenizer, seqlen, batch_size, revision))
        return 7.25, 13

    monkeypatch.setattr(logical_eval, "evaluate_wikitext2", fake_evaluate_wikitext2)
    args = types.SimpleNamespace(
        model="model-path",
        checkpoint="logical.pt",
        wikitext2_revision="wikitext-commit",
    )
    model = object()
    tokenizer = object()

    payload = logical_eval.write_matched_ppl(model, tokenizer, args, tmp_path)

    assert calls == [(model, tokenizer, 2048, 1, "wikitext-commit")]
    assert payload["model"] == "model-path"
    assert payload["checkpoint"] == "logical.pt"
    assert payload["ppl"] == 7.25
    assert payload["sequence_count"] == 13
    assert payload["seqlen"] == 2048
    assert payload["dataset"] == "wikitext2_test"
    assert payload["dataset_revision"] == "wikitext-commit"
    assert payload["ce_implementation"] == (
        "experiments.llama_qtip.evaluate_qtip_checkpoint.evaluate_wikitext2"
    )
    assert payload["completed"] is True
    assert (tmp_path / "matched_ppl.json").exists()


def test_matched_ppl_flag_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_logical_checkpoint_lm_eval.py",
            "--model",
            "m",
            "--checkpoint",
            "c",
            "--output-dir",
            "out",
        ],
    )
    assert logical_eval.parse_args().matched_ppl is False
    assert logical_eval.parse_args().wikitext2_revision == "main"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_logical_checkpoint_lm_eval.py",
            "--model",
            "m",
            "--checkpoint",
            "c",
            "--output-dir",
            "out",
            "--matched-ppl",
            "--wikitext2-revision",
            "wikitext-commit",
        ],
    )
    args = logical_eval.parse_args()
    assert args.matched_ppl is True
    assert args.wikitext2_revision == "wikitext-commit"
