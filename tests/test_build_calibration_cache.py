from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

from experiments.qwen_qtip import build_calibration_cache as builder
from experiments.qwen_qtip.calibration_data import load_qtip_calibration_cache
from scripts.prepare_qtip_baseline_cache import file_sha256, prepare_cache


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_build_and_reuse_tokenizer_specific_cache(tmp_path, monkeypatch, model_type):
    output = tmp_path / "cache.pt"
    args = Namespace(tokenizer_model=str(tmp_path / model_type), output=str(output),
                     hessian_devset_size=2, finetune_devset_size=2, ctx_size=4,
                     seed=0, documents_per_batch=2, dataset_revision="main")
    monkeypatch.setattr(builder, "parse_args", lambda: args)
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained",
                        lambda _: Namespace(model_type=model_type, vocab_size=32))
    tokenizer = Mock(return_value={"input_ids": [[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]]})
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer_loader)

    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, indices):
            return {"text": ["document" for _ in indices]}

    dataset_loader = Mock(return_value=Dataset())
    monkeypatch.setattr(builder, "load_dataset", dataset_loader)
    builder.main()
    raw = torch.load(output, weights_only=False)
    assert raw["metadata"]["tokenizer_model_type"] == model_type
    assert raw["metadata"]["tokenizer_model"] == args.tokenizer_model
    assert raw["finetune_tokens"].dtype == torch.int32
    assert raw["finetune_tokens"].tolist() == [[1, 2, 3, 4], [1, 6, 7, 8]]
    assert raw["hessian_tokens"].tolist() == [[1, 2, 3, 4], [5, 1, 6, 7]]
    tokenizer.assert_called_once_with(["document", "document"], add_special_tokens=True,
                                      truncation=False)
    tokenizer_loader.assert_called_once_with(args.tokenizer_model, use_fast=True)
    dataset_loader.assert_called_once_with(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
        revision="main",
    )
    cache = load_qtip_calibration_cache(output, model_type=model_type, vocab_size=32,
                                       ctx_size=4, hessian_devset_size=2, finetune_devset_size=2)
    assert cache["finetune_tokens"].dtype == torch.int64
    original_bytes = output.read_bytes()
    builder.main()
    assert output.read_bytes() == original_bytes
    assert dataset_loader.call_count == tokenizer_loader.call_count == 1
    with pytest.raises(ValueError, match="tokenizer_model_type mismatch"):
        load_qtip_calibration_cache(output, model_type="llama" if model_type == "qwen3" else "qwen3",
                                    vocab_size=32, ctx_size=4, hessian_devset_size=2,
                                    finetune_devset_size=2)


def test_reject_other_model_family_before_loading_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "parse_args", lambda: Namespace(
        tokenizer_model="unsupported", output=str(tmp_path / "cache.pt")))
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained",
                        lambda _: Namespace(model_type="mistral"))
    tokenizer_loader = Mock()
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer_loader)
    with pytest.raises(ValueError, match="mistral"):
        builder.main()
    tokenizer_loader.assert_not_called()
    assert not (tmp_path / "cache.pt").exists()


def test_baseline_cache_adapter_binds_hessian_alias_and_streaming_source_hash(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(bytes(range(256)) * 32)
    source = {
        "metadata": {
            "tokenizer_model": str(model_path),
            "hessian_devset_size": 2,
            "finetune_devset_size": 2,
            "ctx_size": 4,
            "seed": 0,
        },
        "hessian_tokens": torch.arange(8, dtype=torch.int32).reshape(2, 4),
        "finetune_tokens": torch.arange(8, 16, dtype=torch.int32).reshape(2, 4),
    }
    result = prepare_cache(source, model_path=model_path, source_path=source_path)
    assert torch.equal(result["input_ids"], source["hessian_tokens"])
    assert torch.equal(result["finetune_tokens"], source["finetune_tokens"])
    assert result["metadata"]["reproducibility"]["source_cache_sha256"] == file_sha256(source_path)
