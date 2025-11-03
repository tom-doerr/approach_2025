import numpy as np
from vkb.finetune import FrameDataset


def _mk_color_samples(n=2):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 2, "h": 32, "w": 32}
        # make a simple colored pattern so sat/contrast/wb have effect
        a = np.arange(32, dtype=np.uint8)
        r = np.tile(a[:, None], (1, 32))
        g = np.zeros((32, 32), dtype=np.uint8)
        b = np.tile(a[None, :], (32, 1))
        img = np.dstack([r, g, b]).astype(np.uint8)
        self.arr = np.stack([img for _ in range(2)], 0)


def test_color_jitter_train_only(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    # No jitter baseline
    ds_base = FrameDataset(_mk_color_samples(), img_size=32, data_root='.', mode='train', aug='none', brightness=0.0, warp=0.0, sat=0.0, contrast=0.0, wb=0.0)
    x0, _ = ds_base[0]

    # Force deterministic max factors
    import numpy as _np
    def _u(low, high, size=None):
        return (high if size is None else _np.full(size, high, dtype=float))
    monkeypatch.setattr(_np.random, 'uniform', _u)

    # Train: jitter on → different
    ds_tr = FrameDataset(_mk_color_samples(), img_size=32, data_root='.', mode='train', aug='none', brightness=0.0, warp=0.0, sat=0.1, contrast=0.1, wb=0.1)
    x1, _ = ds_tr[0]
    assert (x0 != x1).any(), 'train jitters should alter pixels'

    # Val: jitters ignored → equal to baseline processing
    ds_val = FrameDataset(_mk_color_samples(), img_size=32, data_root='.', mode='val', aug='none', brightness=0.0, warp=0.0, sat=0.1, contrast=0.1, wb=0.1)
    y1, _ = ds_val[0]
    ds_val0 = FrameDataset(_mk_color_samples(), img_size=32, data_root='.', mode='val', aug='none', brightness=0.0, warp=0.0, sat=0.0, contrast=0.0, wb=0.0)
    y0, _ = ds_val0[0]
    assert np.allclose(y0.numpy(), y1.numpy()), 'val/test should not apply jitters'
