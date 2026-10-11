from types import SimpleNamespace

import pytest
import torch

from src.trainer import QuantizationTrainer
from src.quantization import NOWAG_FORMAT


def _trainer(*, enabled=True, minimum=2):
    trainer = object.__new__(QuantizationTrainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            oom_microbatch_backoff=enabled,
            min_device_microbatch_size=minimum,
        )
    )
    trainer.optimizer = None
    trainer.global_rank = 0
    trainer._clear_failed_step = lambda: None
    return trainer


def test_cuda_oom_retries_same_batch_with_halved_microbatch():
    trainer = _trainer()
    attempts = []

    def train_step(batch, temperature, scale, microbatch_size):
        attempts.append(microbatch_size)
        if microbatch_size > 4:
            raise RuntimeError("CUDA out of memory")
        return 1.25

    trainer.train_step = train_step
    batch = torch.zeros(64, 1, 1)

    loss, selected = trainer._train_step_with_oom_backoff(
        batch, 1.0, 1.0, 16
    )

    assert loss == 1.25
    assert selected == 4
    assert attempts == [16, 8, 4]


def test_non_oom_runtime_error_is_not_swallowed():
    trainer = _trainer()
    trainer.train_step = lambda *args: (_ for _ in ()).throw(
        RuntimeError("shape mismatch")
    )

    with pytest.raises(RuntimeError, match="shape mismatch"):
        trainer._train_step_with_oom_backoff(torch.zeros(8, 1, 1), 1.0, 1.0, 8)


def test_backoff_stops_at_configured_minimum():
    trainer = _trainer(minimum=4)
    attempts = []

    def always_oom(batch, temperature, scale, microbatch_size):
        attempts.append(microbatch_size)
        raise RuntimeError("CUDA out of memory")

    trainer.train_step = always_oom

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        trainer._train_step_with_oom_backoff(torch.zeros(64, 1, 1), 1.0, 1.0, 16)

    assert attempts == [16, 8, 4]


def test_backoff_can_be_disabled_for_strict_reproduction():
    trainer = _trainer(enabled=False)
    trainer.train_step = lambda *args: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory")
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        trainer._train_step_with_oom_backoff(torch.zeros(16, 1, 1), 1.0, 1.0, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_hard_state_snapshot_is_offloaded_to_cpu():
    state = {
        "format": NOWAG_FORMAT,
        "weight": torch.arange(4, device="cuda", dtype=torch.float32),
        "groupsize": 128,
    }

    cloned = QuantizationTrainer._clone_hard_state(state)

    assert cloned["weight"].device.type == "cpu"
    assert torch.equal(cloned["weight"], torch.arange(4, dtype=torch.float32))
    assert cloned["groupsize"] == 128
