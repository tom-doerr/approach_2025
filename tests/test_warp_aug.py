import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=2):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 2, "h": 32, "w": 32}
        # diagonal gradient so warp changes pixels
        a = np.arange(32, dtype=np.uint8)
        img = (a[:, None] + a[None, :]) % 255
        self.arr = np.stack([np.dstack([img, img, img]) for _ in range(2)], 0)


def test_warp_changes_pixels(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    ds0 = FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='train', aug='none', warp=0.0)
    x0, _ = ds0[0]

    ds1 = FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='train', aug='none', warp=0.2)
    x1, _ = ds1[0]

    assert (x0 != x1).any(), 'warp>0 should alter pixels vs. warp=0'

