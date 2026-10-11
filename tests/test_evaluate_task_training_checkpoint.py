from experiments.nowag_d6.evaluate_task_training_checkpoint import candidate_passes_internal_gate


def test_internal_gate_requires_macro_non_regression_and_balanced_loss_gain() -> None:
    base = {"macro_accuracy": 0.6, "balanced_loss": 0.9}
    assert candidate_passes_internal_gate(
        base, {"macro_accuracy": 0.6, "balanced_loss": 0.8}
    )
    assert candidate_passes_internal_gate(
        base, {"macro_accuracy": 0.61, "balanced_loss": 0.89}
    )
    assert not candidate_passes_internal_gate(
        base, {"macro_accuracy": 0.59, "balanced_loss": 0.8}
    )
    assert not candidate_passes_internal_gate(
        base, {"macro_accuracy": 0.61, "balanced_loss": 0.91}
    )
