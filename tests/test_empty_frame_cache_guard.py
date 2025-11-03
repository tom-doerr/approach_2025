import numpy as np


def test_empty_frame_cache_raises(monkeypatch):
    from vkb.dataset import FrameDataset
    from vkb import cache as vcache
    # Fake memmap with zero frames
    arr = np.zeros((0, 4, 5, 3), dtype=np.uint8)
    meta = {"n": 0, "h": 4, "w": 5}
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (arr, meta))
    ds = FrameDataset([("vid", 0, 0)], img_size=8, data_root='.', mode='train')
    import pytest
    with pytest.raises(RuntimeError):
        _ = ds[0]

