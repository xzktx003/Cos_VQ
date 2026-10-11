"""Event-driven handoff from scalar GSQ to the one-GPU Vector-GSQ run."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import select
import shutil


SYS_PIDFD_OPEN = 434


def _pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = int(libc.syscall(SYS_PIDFD_OPEN, int(pid), 0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def validate_scalar_ppl(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("completed"):
        raise ValueError("Scalar GSQ PPL is not completed")
    if int(payload.get("quantized_prefix_layers", -1)) != 32:
        raise ValueError("Scalar GSQ PPL must cover all 32 layers")
    if payload.get("dataset") != "wikitext2_test":
        raise ValueError("Scalar GSQ PPL must use WikiText2 test")
    if int(payload.get("ppl_seqlength", -1)) != 2048:
        raise ValueError("Scalar GSQ PPL must use seqlength 2048")
    ppl = float(payload.get("ppl", float("nan")))
    if not math.isfinite(ppl) or ppl <= 0:
        raise ValueError("Scalar GSQ PPL must be finite and positive")
    return payload


def wait_for_process_exit(pid):
    """Block on Linux pidfd; this is event-driven and performs no polling."""
    try:
        descriptor = _pidfd_open(int(pid))
    except OSError as error:
        if error.errno == errno.ESRCH:
            return
        raise
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        poller.poll()
    finally:
        os.close(descriptor)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--scalar-ppl", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--cuda-physical-device", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=45000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pid <= 0 or args.cuda_physical_device < 0 or args.min_free_mib < 0:
        raise ValueError("pid, GPU index, and memory threshold must be non-negative")
    launcher = Path(args.launcher).resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"Vector launcher not found: {launcher}")

    print(f"Waiting event-driven for scalar process {args.pid}", flush=True)
    wait_for_process_exit(args.pid)
    scalar = validate_scalar_ppl(args.scalar_ppl)
    print(
        f"Scalar completed at PPL={scalar['ppl']}; launching Vector-GSQ on "
        f"physical GPU {args.cuda_physical_device}",
        flush=True,
    )
    environment = os.environ.copy()
    environment["CUDA_PHYSICAL_DEVICE"] = str(args.cuda_physical_device)
    environment["MIN_FREE_MIB"] = str(args.min_free_mib)
    bash = shutil.which("bash")
    if bash is None:
        raise FileNotFoundError("bash is not available")
    os.execve(bash, [bash, str(launcher)], environment)


if __name__ == "__main__":
    main()
