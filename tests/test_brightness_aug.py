import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=2):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 2, "h": 8, "w": 8}
        self.arr = np.full((2, 8, 8, 3), 100, dtype=np.uint8)


def test_brightness_aug_applies(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    # Make uniform draw deterministic and positive (always +b)
    monkeypatch.setattr(np.random, 'uniform', lambda lo, hi: hi)

    ds = FrameDataset(_mk_samples(), img_size=8, data_root='.', mode='train', aug='none', brightness=0.2)
    x, _ = ds[0]
    ds0 = FrameDataset(_mk_samples(), img_size=8, data_root='.', mode='train', aug='none', brightness=0.0)
    x0, _ = ds0[0]
    assert (x != x0).any()

