import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=2):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 2, "h": 16, "w": 16}
        # diagonal gradient
        a = np.arange(16, dtype=np.uint8)
        img = (a[:, None] + a[None, :]) % 255
        self.arr = np.stack([np.dstack([img, img, img]) for _ in range(2)], 0)


def test_rot_deg_changes_pixels(monkeypatch):
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))
    # Monkeypatch rotate to a simple pixel roll so the test doesn't need cv2
    import vkb.finetune as ft
    monkeypatch.setattr(ft, '_rotate_square', lambda img, ang: np.roll(img, 1, axis=0))

    ds0 = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', rot_deg=0.0)
    x0, _ = ds0[0]
    ds1 = FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', rot_deg=5.0)
    x1, _ = ds1[0]
    assert (x0 != x1).any(), 'rot_deg>0 should alter pixels vs rot_deg=0'

