import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.llama_qtip.summarize_structure_aware_vs_original_gsq import (  # noqa: E402
    compare_gate_results,
)


def _vector_summary():
    return {
        "model": "local-llama",
        "dataset": {"ppl": "WikiText2 test", "ppl_seqlength": 2048},
        "rate_fair_candidate": {
            "method": "joint-group256",
            "logical_bpp": 2.08199,
            "prefix1_ppl": 6.20,
        },
    }


def _original_result():
    return {
        "completed": True,
        "model": "local-llama",
        "dataset": "wikitext2_test",
        "ppl_seqlength": 2048,
        "quantized_prefix_layers": 1,
        "ppl": 6.25,
    }


def test_rate_and_quality_gate_requires_both_improvements():
    result = compare_gate_results(_vector_summary(), _original_result(), 2.125)

    assert result["protocol_matched"] is True
    assert result["vector_rate_not_higher"] is True
    assert result["vector_ppl_better"] is True
    assert result["vector_exceeds_original_gsq"] is True
    assert result["ppl_delta_vector_minus_original"] == pytest.approx(-0.05)


def test_lower_ppl_cannot_pass_when_vector_rate_is_higher():
    vector = _vector_summary()
    vector["rate_fair_candidate"]["logical_bpp"] = 2.20
    result = compare_gate_results(vector, _original_result(), 2.125)
    assert result["vector_ppl_better"] is True
    assert result["vector_rate_not_higher"] is False
    assert result["vector_exceeds_original_gsq"] is False


def test_protocol_mismatch_is_rejected():
    original = _original_result()
    original["ppl_seqlength"] = 4096
    with pytest.raises(ValueError, match="2048"):
        compare_gate_results(_vector_summary(), original, 2.125)
