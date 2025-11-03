import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=1):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 1, "h": 16, "w": 16}
        img = np.full((16,16,3), 128, dtype=np.uint8)
        self.arr = np.stack([img], 0)


def test_random_erasing_train_only(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    # Make erase always trigger and deterministic
    import numpy as _np
    # First uniform for p, then for area and aspect; choose fixed mid values
    calls = {'i': 0}
    def _u(a, b):
        i = calls['i']; calls['i'] += 1
        if i == 0:  # p
            return 0.0  # less than erase_p (we set erase_p=1 so unused)
        return (a + b) / 2.0
    monkeypatch.setattr(_np.random, 'uniform', _u)
    monkeypatch.setattr(_np.random, 'normal', lambda mu, sigma, size=None: _np.zeros(size, dtype=_np.float32))

    ds = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', erase_p=1.0)
    x1, _ = ds[0]

    ds_no = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', erase_p=0.0)
    x0, _ = ds_no[0]
    assert (x0 != x1).any(), 'erasing should alter pixels vs baseline when erase_p=1'

def test_random_erasing_ignored_in_val(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))

    ds_v = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='val', aug='none', erase_p=1.0)
    ds_p = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='val', aug='none', erase_p=0.0)
    xv, _ = ds_v[0]
    xp, _ = ds_p[0]
    assert np.allclose(xv.numpy(), xp.numpy()), 'validation must not apply random erasing'

