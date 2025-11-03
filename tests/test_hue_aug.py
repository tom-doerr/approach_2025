import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=2):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 2, "h": 32, "w": 32}
        # colored gradient ensures hue rotation changes pixels
        a = np.arange(32, dtype=np.uint8)
        r = np.tile(a[:, None], (1, 32))
        g = np.zeros((32, 32), dtype=np.uint8)
        b = np.tile(a[None, :], (32, 1))
        img = np.dstack([r, g, b]).astype(np.uint8)
        self.arr = np.stack([img for _ in range(2)], 0)


def test_hue_shift_changes_pixels(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    ds0 = FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='train', aug='none', brightness=0.0, warp=0.0, hue=0.0)
    x0, _ = ds0[0]

    import numpy as _np
    def _u(low, high, size=None):
        return (high if size is None else _np.full(size, high, dtype=float))
    monkeypatch.setattr(_np.random, 'uniform', _u)

    ds1 = FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='train', aug='none', brightness=0.0, warp=0.0, hue=0.1)
    x1, _ = ds1[0]
    assert (x0 != x1).any(), 'hue>0 should alter pixels vs. hue=0'

