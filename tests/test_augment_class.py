import numpy as np


def _grad_img(w=16, h=16):
    a = np.arange(h, dtype=np.uint8)[:, None]
    b = np.arange(w, dtype=np.uint8)[None, :]
    g = (a + b) % 255
    return np.dstack([g, g, g])


def test_augment_exists_and_val_ignores_aug(monkeypatch):
    import vkb.augment as ft
    # minimal cv2 stub to satisfy resize
    import sys
    class _CV:
        def resize(self, img, sz):
            H, W = sz[1], sz[0]
            ih, iw = img.shape[:2]
            ys = (np.linspace(0, ih - 1, H)).astype(int)
            xs = (np.linspace(0, iw - 1, W)).astype(int)
            return img[ys][:, xs]
    monkeypatch.setitem(sys.modules, 'cv2', _CV())

    assert hasattr(ft, 'Augment')
    img = _grad_img(16, 16)
    a0 = ft.Augment(size=24, mode='val', aug='rot360', warp=0.5)
    out = a0.apply(img)
    assert out.shape[:2] == (24, 24)

    # val path should ignore warp; same output for warp=0 and 0.5
    a1 = ft.Augment(size=24, mode='val', aug='rot360', warp=0.0)
    out2 = a1.apply(img)
    assert (out == out2).all()


def test_augment_train_warp_changes_pixels(monkeypatch):
    import vkb.augment as ft
    # stub helpers: zoom to size, rotate identity, warp as right-shift by 1
    def _nn_resize(img, out_size, *_):
        ih, iw = img.shape[:2]
        ys = (np.linspace(0, ih - 1, out_size)).astype(int)
        xs = (np.linspace(0, iw - 1, out_size)).astype(int)
        return img[ys][:, xs]
    monkeypatch.setattr(ft, '_zoom_out', _nn_resize)
    monkeypatch.setattr(ft, '_rotate_square', lambda img, ang: img)
    monkeypatch.setattr(ft, '_perspective_warp_inward', lambda img, frac: np.pad(img[:, :-1, :], ((0,0),(1,0),(0,0))))
    # minimal cv2 stub for resize path
    import sys
    class _CV:
        def resize(self, img, sz):
            H, W = sz[1], sz[0]
            ih, iw = img.shape[:2]
            ys = (np.linspace(0, ih - 1, H)).astype(int)
            xs = (np.linspace(0, iw - 1, W)).astype(int)
            return img[ys][:, xs]
    monkeypatch.setitem(sys.modules, 'cv2', _CV())

    img = _grad_img(20, 20)
    a_no = ft.Augment(size=24, mode='train', aug='rot360', warp=0.0)
    a_w = ft.Augment(size=24, mode='train', aug='rot360', warp=0.2)
    out_no = a_no.apply(img)
    out_w = a_w.apply(img)
    assert out_no.shape == out_w.shape == (24, 24, 3)
    assert (out_no != out_w).any()
