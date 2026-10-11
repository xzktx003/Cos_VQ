from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.llama_qtip.summarize_full_structure_aware_vs_original_gsq import (  # noqa: E402
    compare_full_results,
)


MODEL = "/models/Meta-Llama-3-8B"


def _vector_quantization():
    return {
        "completed": True,
        "arguments": {
            "model": MODEL,
            "checkpoint": "/results/vector/checkpoint.pt",
            "max_layers": 32,
            "seed": 0,
            "stage": "gsq",
            "gsq_objective": "staged",
            "staged_joint_group_scales": True,
            "scale_group_size": 160,
            "scale_finetune_steps": 0,
            "block_polish_steps": 0,
            "vector_dim": 6,
            "bits_per_weight": 2,
        },
        "logical_effective_bpp": 2.1207557091346154,
        "layers": [
            {
                "layer": index,
                "gsq": {"teacher_reference": "original_dense_network"},
            }
            for index in range(32)
        ],
    }


def _vector_ppl():
    return {
        "completed": True,
        "arguments": {
            "model": MODEL,
            "checkpoint": "/results/vector/checkpoint.pt",
            "ppl_seqlen": 2048,
            "ppl_split": "test",
        },
        "quantized_layers": 32,
        "quantized_linears": 224,
        "expected_linears": 224,
        "all_fresh_reconstructions_exact": True,
        "logical_effective_bpp": 2.1207557091346154,
        "ppl": 8.0,
        "ppl_split": "test",
    }


def _original_ppl():
    return {
        "completed": True,
        "model": MODEL,
        "dataset": "wikitext2_test",
        "ppl_seqlength": 2048,
        "quantized_prefix_layers": 32,
        "ppl": 8.5,
    }


def test_full_comparator_requires_rate_and_ppl_improvement():
    result = compare_full_results(
        _vector_quantization(), _vector_ppl(), _original_ppl(), 2.125
    )

    assert result["protocol_matched"] is True
    assert result["quantized_layers"] == 32
    assert result["vector_rate_not_higher"] is True
    assert result["vector_ppl_better"] is True
    assert result["vector_exceeds_original_gsq"] is True
    assert result["ppl_delta_vector_minus_original"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda q, p, o: p.update(quantized_layers=31), "32"),
        (lambda q, p, o: p.update(all_fresh_reconstructions_exact=False), "reconstruction"),
        (lambda q, p, o: p["arguments"].update(ppl_seqlen=4096), "2048"),
        (lambda q, p, o: o.update(model="/models/other"), "model"),
    ],
)
def test_full_comparator_rejects_protocol_mismatches(mutation, message):
    quantization = _vector_quantization()
    vector_ppl = _vector_ppl()
    original = _original_ppl()
    mutation(quantization, vector_ppl, original)

    with pytest.raises(ValueError, match=message):
        compare_full_results(quantization, vector_ppl, original, 2.125)


def test_full_comparator_cannot_pass_at_higher_rate():
    quantization = _vector_quantization()
    vector_ppl = _vector_ppl()
    quantization["logical_effective_bpp"] = 2.13
    vector_ppl["logical_effective_bpp"] = 2.13

    result = compare_full_results(quantization, vector_ppl, _original_ppl(), 2.125)

    assert result["vector_ppl_better"] is True
    assert result["vector_rate_not_higher"] is False
    assert result["vector_exceeds_original_gsq"] is False


def test_full_comparator_rejects_a_different_scale_group_method():
    quantization = _vector_quantization()
    quantization["arguments"]["scale_group_size"] = 128

    with pytest.raises(ValueError, match="group160"):
        compare_full_results(
            quantization,
            _vector_ppl(),
            _original_ppl(),
            2.125,
        )


def test_full_comparator_rejects_hybrid_stage_teachers():
    quantization = _vector_quantization()
    quantization["layers"][17]["gsq"]["teacher_reference"] = (
        "quantized_upstream_hybrid"
    )

    with pytest.raises(ValueError, match="dense teacher"):
        compare_full_results(
            quantization,
            _vector_ppl(),
            _original_ppl(),
            2.125,
        )
