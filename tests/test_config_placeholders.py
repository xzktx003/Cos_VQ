"""Sanity checks for src.config.load_config: ${VAR} expansion and GSQ_* / WANDB_* overrides."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402


def _write_cfg(text: str):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        f.write(text)
        f.flush()
        return f.name
    finally:
        f.close()


def main():
    tdir = tempfile.mkdtemp(prefix="gsq_cfg_test_")
    ev_test = os.environ.copy()

    # 1) expandvars on load
    p1 = _write_cfg(
        f"""
model:
  name: "${{GSQ_TEST_ROOT}}/models/x"
training:
  checkpoint_dir: "${{GSQ_TEST_ROOT}}/ckpt"
  log_dir: "${{GSQ_TEST_ROOT}}/logs"
wandb: false
"""
    )
    try:
        os.environ["GSQ_TEST_ROOT"] = tdir
        c1 = load_config(p1)
        assert c1.model.name == f"{tdir}/models/x", c1.model.name
        assert c1.training.checkpoint_dir == f"{tdir}/ckpt"
        assert c1.training.log_dir == f"{tdir}/logs"
    finally:
        os.environ.clear()
        os.environ.update(ev_test)
        os.unlink(p1)

    # 2) GSQ_* env overrides win over YAML (after YAML expansion)
    p2 = _write_cfg(
        """
model:
  name: "/yaml/model"
training:
  checkpoint_dir: "/yaml/ckpt"
  log_dir: "/yaml/logs"
  act_cache_dir: "/yaml/act"
wandb:
  enabled: false
  project: "yaml-proj"
  entity: "yaml-ent"
"""
    )
    try:
        os.environ["GSQ_MODEL_NAME"] = "/env/model"
        os.environ["GSQ_CHECKPOINT_DIR"] = "/env/ckpt"
        os.environ["GSQ_LOG_DIR"] = "/env/logs"
        os.environ["GSQ_ACT_CACHE_DIR"] = "/env/act"
        os.environ["GSQ_ACT_CACHE_MMAP_THRESHOLD_GB"] = "2.5"
        os.environ["GSQ_DEVICE_MICROBATCH_SIZE"] = "64"
        os.environ["GSQ_ACTIVATION_MICROBATCH_SIZE"] = "32"
        c2 = load_config(p2)
        assert c2.model.name == "/env/model"
        assert c2.training.checkpoint_dir == "/env/ckpt"
        assert c2.training.log_dir == "/env/logs"
        assert c2.training.act_cache_dir == "/env/act"
        assert c2.training.act_cache_mmap_threshold_gb == 2.5
        assert c2.training.device_microbatch_size == 64
        assert c2.training.activation_microbatch_size == 32
    finally:
        os.environ.clear()
        os.environ.update(ev_test)
        os.unlink(p2)

    # 3) WANDB_* overrides YAML project/entity
    p3 = _write_cfg(
        """
model:
  name: "x"
training:
  checkpoint_dir: "c"
  log_dir: "l"
wandb:
  enabled: true
  project: "from-yaml"
  entity: "ent-yaml"
"""
    )
    try:
        os.environ["WANDB_PROJECT"] = "from-env"
        os.environ["WANDB_ENTITY"] = "ent-env"
        c3 = load_config(p3)
        assert c3.wandb.project == "from-env"
        assert c3.wandb.entity == "ent-env"
    finally:
        os.environ.clear()
        os.environ.update(ev_test)
        os.unlink(p3)

    print("OK: config placeholder expansion + env overrides")


if __name__ == "__main__":
    main()
