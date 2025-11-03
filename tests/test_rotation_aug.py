import numpy as np
from vkb.finetune import FrameDataset


def _mk_samples(n=4):
    return [("fake_vid", i, 0) for i in range(n)]


class _MM:
    def __init__(self):
        self.meta = {"n": 4, "h": 32, "w": 32}
        self.arr = np.stack([
            np.full((32,32,3), 80, dtype=np.uint8) for _ in range(4)
        ], 0)


def test_rot360_shapes(monkeypatch):
    # Use real cv2; patch only cache
    from vkb import cache as vcache
    mm = _MM()
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm.arr, mm.meta))
    # Avoid cv2 dependency: stub zoom and rotate to pure numpy
    import vkb.finetune as ft
    def _nn_resize(img, w, h):
        ih, iw = img.shape[:2]
        ys = (np.linspace(0, ih - 1, h)).astype(int)
        xs = (np.linspace(0, iw - 1, w)).astype(int)
        return img[ys][:, xs]
    monkeypatch.setattr(ft, '_zoom_out', lambda img, out_size, a, b: _nn_resize(img, out_size, out_size))
    monkeypatch.setattr(ft, '_rotate_square', lambda img, ang: img)
    # Minimal cv2 stub for cvtColor used after aug
    import sys
    class _CV:
        def cvtColor(self, img, code): return img
        COLOR_BGR2RGB = 0
    monkeypatch.setitem(sys.modules, 'cv2', _CV())

    ds = FrameDataset(_mk_samples(), img_size=24, data_root='.', mode='train', aug='rot360')
    x, y = ds[0]
    assert tuple(x.shape) == (3, 24, 24)
    assert int(y.item()) == 0
