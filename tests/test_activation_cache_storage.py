from pathlib import Path

import torch

from main import _cleanup_act_cache_mmap, create_dict


def test_activation_cache_stays_in_ram_below_threshold(tmp_path):
    data = create_dict(
        num_samples=4,
        seqlen=8,
        hidden_size=16,
        dtype=torch.bfloat16,
        mmap_dir=str(tmp_path),
        mmap_threshold_bytes=10_000,
    )

    assert "_mmap_path" not in data
    assert data["input"].shape == (4, 8, 16)
    assert not list(tmp_path.iterdir())


def test_activation_cache_uses_and_cleans_mmap_above_threshold(tmp_path):
    data = create_dict(
        num_samples=4,
        seqlen=8,
        hidden_size=16,
        dtype=torch.bfloat16,
        mmap_dir=str(tmp_path),
        mmap_threshold_bytes=1,
    )
    mmap_path = Path(data["_mmap_path"])

    assert mmap_path.exists()
    _cleanup_act_cache_mmap(data)
    assert not mmap_path.exists()
