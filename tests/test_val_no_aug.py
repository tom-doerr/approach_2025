import numpy as np


def _mk_samples():
    return [("/fake/video.mp4", 0, 1)]


def test_validation_has_no_augmentation(monkeypatch):
    from vkb import finetune as ft

    # Fake a deterministic single frame (16x16 RGB) and stub cache I/O
    frame = np.tile(np.arange(0, 16, dtype=np.uint8)[:, None], (1, 16))
    frame = np.stack([frame, frame, frame], axis=2)  # shape HWC
    mm = np.stack([frame], axis=0)

    def fake_ensure(_root, _vp):
        return None

    def fake_open(_root, _vp):
        return mm, {"n": 1}

    import vkb.cache as vcache
    monkeypatch.setattr(vcache, "open_frames_memmap", fake_open, raising=False)
    monkeypatch.setattr(vcache, "ensure_frames_cached", fake_ensure, raising=False)

    # Stub cv2 so resize/cvtColor are identity for 16x16
    class FakeCv2:
        COLOR_BGR2RGB = 0
        def resize(self, img, sz):
            return img
        def cvtColor(self, img, code):
            return img

    import sys
    old_cv2 = sys.modules.get('cv2')
    sys.modules['cv2'] = FakeCv2()
    try:
        samples = _mk_samples()
        # Even if we pass aug/noise/brightness/warp, val mode should ignore them
        ds_val_aug = ft.FrameDataset(samples, img_size=16, data_root='.', mode='val', aug='rot360', noise_std=0.5, brightness=0.5, warp=0.5)
        ds_val_plain = ft.FrameDataset(samples, img_size=16, data_root='.', mode='val', aug='none', noise_std=0.0, brightness=0.0, warp=0.0)
        x_aug, y_aug = ds_val_aug[0]
        x_plain, y_plain = ds_val_plain[0]
        assert y_aug.item() == y_plain.item()
        # Exact equality given our identity stubs
        assert (x_aug == x_plain).all(), "Validation must not apply augmentation/noise/brightness/warp"
    finally:
        if old_cv2 is not None:
            sys.modules['cv2'] = old_cv2
        else:
            sys.modules.pop('cv2', None)
