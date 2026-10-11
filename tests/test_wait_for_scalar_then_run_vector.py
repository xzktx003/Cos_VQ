import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.llama_qtip.wait_for_scalar_then_run_vector import (  # noqa: E402
    validate_scalar_ppl,
)


def _write_scalar_result(path, **overrides):
    payload = {
        "completed": True,
        "quantized_prefix_layers": 32,
        "dataset": "wikitext2_test",
        "ppl_seqlength": 2048,
        "ppl": 8.5,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_supervisor_waits_for_pid_then_executes_vector_launcher(tmp_path):
    scalar_result = tmp_path / "ppl_exact.json"
    _write_scalar_result(scalar_result)
    marker = tmp_path / "launched.txt"
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s %s\\n' \"$CUDA_PHYSICAL_DEVICE\" \"$MIN_FREE_MIB\" > {marker}\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    waited = subprocess.Popen(["sleep", "0.1"])

    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "experiments/llama_qtip/wait_for_scalar_then_run_vector.py"
            ),
            "--pid",
            str(waited.pid),
            "--scalar-ppl",
            str(scalar_result),
            "--launcher",
            str(launcher),
            "--cuda-physical-device",
            "6",
            "--min-free-mib",
            "45000",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8").strip() == "6 45000"


def test_scalar_result_validation_rejects_incomplete_protocol(tmp_path):
    scalar_result = tmp_path / "ppl_exact.json"
    _write_scalar_result(scalar_result, quantized_prefix_layers=31)

    with pytest.raises(ValueError, match="32"):
        validate_scalar_ppl(scalar_result)
