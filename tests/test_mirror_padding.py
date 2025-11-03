import numpy as np


def _mk_samples(n=1):
    return [("fake_vid", 0, 0)]


class _MM:
    def __init__(self, val=80, h=16, w=16):
        self.meta = {"n": 1, "h": h, "w": w}
        self.arr = np.stack([np.full((h,w,3), val, dtype=np.uint8)], 0)


def test_zoomout_uses_mirror_padding(monkeypatch):
    # Force a strong zoom-out (scale=0.5) so padding is present
    import vkb.finetune as ft
    monkeypatch.setattr(__import__('random'), 'uniform', lambda a, b: 0.5)
    def _nn_resize(img, w, h):
        ih, iw = img.shape[:2]
        ys = (np.linspace(0, ih - 1, h)).astype(int)
        xs = (np.linspace(0, iw - 1, w)).astype(int)
        return img[ys][:, xs]
    monkeypatch.setattr(ft, '_resize_to', _nn_resize)
    # Minimal cv2 stub for copyMakeBorder
    class _CV:
        BORDER_REFLECT_101 = 0
        @staticmethod
        def copyMakeBorder(img, top, bottom, left, right, borderType):
            return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='reflect')
    import sys
    monkeypatch.setitem(sys.modules, 'cv2', _CV)

    img = np.full((16,16,3), 80, dtype=np.uint8)
    out = ft._zoom_out(img, out_size=24, scale_min=0.5, scale_max=0.5)
    # Corners should match the center value on uniform input (no black padding)
    center = out[12,12].mean(); corner = out[0,0].mean()
    assert abs(float(center) - float(corner)) < 1.0
