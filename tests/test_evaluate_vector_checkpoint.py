import pytest
from unittest.mock import Mock

from experiments.nowag_d6.evaluate_vector_checkpoint import (
    evaluate_wikitext2_split,
    checkpoint_ppl_experiment_name,
    load_logical_checkpoint,
)


def test_checkpoint_ppl_experiment_name_uses_runtime_model() -> None:
    assert (
        checkpoint_ppl_experiment_name("/models/Qwen3-4B")
        == "qwen3_4b_nowag_d6_checkpoint_independent_ppl"
    )
    assert "06b" not in checkpoint_ppl_experiment_name("/models/Qwen3-32B")


def test_trusted_checkpoint_uses_legacy_compatible_loader() -> None:
    torch_load = Mock(return_value={"layers": {}})

    checkpoint = load_logical_checkpoint(
        "/results/full4096.pt",
        trusted=True,
        torch_load=torch_load,
    )

    assert checkpoint == {"layers": {}}
    torch_load.assert_called_once_with(
        "/results/full4096.pt",
        map_location="cpu",
        weights_only=False,
    )


def test_untrusted_checkpoint_keeps_safe_loader() -> None:
    torch_load = Mock(return_value={"layers": {}})

    load_logical_checkpoint(
        "/results/full4096.pt",
        trusted=False,
        torch_load=torch_load,
    )

    torch_load.assert_called_once_with(
        "/results/full4096.pt",
        map_location="cpu",
        weights_only=True,
    )


def test_wikitext_split_rejects_unknown_split_without_loading_data() -> None:
    with pytest.raises(ValueError, match="unsupported WikiText2 split"):
        evaluate_wikitext2_split(
            object(),
            object(),
            split="unknown",
            sequence_length=2048,
            batch_size=8,
        )
