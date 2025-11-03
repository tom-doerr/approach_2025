import io, sys, types, numpy as np


def test_img_size_flag_passes_to_dataset(monkeypatch):
    import train_frames as tf
    import vkb.finetune as ft

    # Stub cache I/O with a tiny constant image
    import vkb.cache as vcache
    mm = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm, {"n": 1, "h": 8, "w": 8}))

    # Avoid importing real cv2; use a minimal stub for resize/cvtColor
    class FakeCv2:
        COLOR_BGR2RGB = 0
        def resize(self, img, sz):
            # return a simple nearest-like expansion by tiling
            H, W = sz[1], sz[0]
            import numpy as _np
            ys = (_np.linspace(0, img.shape[0]-1, H)).astype(int)
            xs = (_np.linspace(0, img.shape[1]-1, W)).astype(int)
            return img[ys][:, xs]
        def cvtColor(self, img, code):
            return img
    import sys as _sys
    old = _sys.modules.get('cv2'); _sys.modules['cv2'] = FakeCv2()
    try:
        args = tf.parse_args(["--clf","dl","--device","cpu","--epochs","1","--batch-size","2","--img-size","32","--aug","none","--no-dynamic-aug"])
        samples = [("vid.mp4", 0, 0)]
        tr_idx, va_idx = [0], []
        ft._make_loaders(args, samples, tr_idx, va_idx, n_classes=1, device='cpu')
        ds = getattr(args, '_ds_tr', None)
        x, y = ds[0]
        assert tuple(x.shape[-2:]) == (32, 32)
    finally:
        if old is not None:
            _sys.modules['cv2'] = old
        else:
            _sys.modules.pop('cv2', None)
