import sys
import inspect
import threading
import time
import types
from pathlib import Path


if "lion_pytorch" not in sys.modules:
    sys.modules["lion_pytorch"] = types.SimpleNamespace(Lion=object)

from experiments.nowag_d6.run_prefix_vector_gsq import (
    _run_bounded_linear_jobs,
    _run_vector_gptq_jobs,
    _warm_vector_gptq_cuda_linalg,
    parse_args,
)


def test_vector_gptq_defaults_to_top128_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_prefix_vector_gsq.py",
            "--token-cache",
            "tokens.pt",
            "--output",
            "result.json",
            "--checkpoint",
            "checkpoint.pt",
            "--stage",
            "gptq",
        ],
    )

    assert parse_args().gptq_candidates == 128


def test_bounded_linear_jobs_overlap_without_exceeding_limit() -> None:
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lock = threading.Lock()
    two_jobs_started = threading.Event()
    active = 0
    peak_active = 0

    def worker(name: str) -> str:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            if active == 2:
                two_jobs_started.set()
        assert two_jobs_started.wait(timeout=1.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return f"done:{name}"

    results = _run_bounded_linear_jobs(names, worker, concurrency=2)

    assert peak_active == 2
    assert list(results) == names
    assert results == {name: f"done:{name}" for name in names}


def test_bounded_linear_jobs_keep_serial_execution_in_caller_thread() -> None:
    caller_thread = threading.get_ident()

    results = _run_bounded_linear_jobs(
        ["q_proj", "k_proj"],
        lambda name: (name, threading.get_ident()),
        concurrency=1,
    )

    assert results == {
        "q_proj": ("q_proj", caller_thread),
        "k_proj": ("k_proj", caller_thread),
    }


def test_bounded_linear_jobs_reject_invalid_concurrency() -> None:
    try:
        _run_bounded_linear_jobs(["q_proj"], lambda name: name, concurrency=0)
    except ValueError as error:
        assert "concurrency" in str(error)
    else:
        raise AssertionError("expected invalid concurrency to fail")


def test_parallel_gptq_warms_cuda_linalg_before_worker_submission() -> None:
    source = inspect.getsource(_run_vector_gptq_jobs)

    warmup = source.index("_warm_vector_gptq_cuda_linalg(device)")
    ready_event = source.index("ready = torch.cuda.Event()")
    submission = source.index("_run_bounded_linear_jobs(names, cuda_worker")

    assert warmup < ready_event < submission


def test_cuda_linalg_warmup_covers_all_gptq_factorizations() -> None:
    source = inspect.getsource(_warm_vector_gptq_cuda_linalg)

    assert "torch.linalg.cholesky" in source
    assert "torch.cholesky_inverse" in source
    assert "upper=True" in source

