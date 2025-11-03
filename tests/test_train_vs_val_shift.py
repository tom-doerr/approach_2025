def _mk_samples():
    return [("vid", 0, 0)]


def test_train_vs_val_shift_diff(monkeypatch):
    import types, numpy as np
    import vkb.dataset as d

    # Stub cache to return a tiny synthetic frame
    H = W = 16
    frame = np.zeros((H, W, 3), dtype=np.uint8); frame[H//2, W//2] = [255, 255, 255]
    monkeypatch.setattr('vkb.cache.ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr('vkb.cache.open_frames_memmap', lambda *a, **k: (np.asarray([frame]), {"n": 1}))

    # Deterministic large shift
    monkeypatch.setattr('numpy.random.uniform', lambda *a, **k: 1.0)

    # Minimal cv2 stub with warpAffine that shifts via roll (sufficient to change pixels)
    class _CV:
        COLOR_BGR2RGB = 1
        INTER_LINEAR = 1
        BORDER_REFLECT_101 = 1
        def cvtColor(self, img, code):
            return img[:, :, ::-1]
        def warpAffine(self, img, M, size, flags=None, borderMode=None):
            return np.roll(img, shift=1, axis=1)
    monkeypatch.setitem(__import__('sys').modules, 'cv2', _CV())

    ds_tr = d.FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', shift=0.2,
                           brightness=0.0, warp=0.0, sat=0.0, contrast=0.0, wb=0.0, hue=0.0, rot_deg=0.0)
    ds_va = d.FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='val', aug='none', shift=0.0,
                           brightness=0.0, warp=0.0, sat=0.0, contrast=0.0, wb=0.0, hue=0.0, rot_deg=0.0)

    x_tr, _ = ds_tr[0]
    x_va, _ = ds_va[0]
    assert (x_tr != x_va).any(), 'shift should alter training sample vs validation'

