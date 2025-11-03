import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=1):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 1, "h": 16, "w": 16}
        img = np.full((16,16,3), 128, dtype=np.uint8)
        self.arr = np.stack([img], 0)


def test_noise_varies_by_sample(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    # Baseline (no noise)
    ds0 = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', noise_std=0.0)
    x0, _ = ds0[0]

    # Force max sigma and deterministic noise values
    import numpy as _np
    monkeypatch.setattr(_np.random, 'uniform', lambda a, b: b)
    monkeypatch.setattr(_np.random, 'normal', lambda mu, sigma, size=None: _np.full(size, sigma, dtype=_np.float32))

    ds = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', noise_std=0.2)
    x1, _ = ds[0]
    assert (x0 != x1).any(), 'noise_std>0 should alter pixels vs baseline'

    # Force zero sigma → identical to baseline
    monkeypatch.setattr(_np.random, 'uniform', lambda a, b: 0.0)
    x2, _ = ds[0]
    # normal() still returns sigma values, but sigma=0 → zeros
    assert np.allclose(x0.numpy(), x2.numpy())
