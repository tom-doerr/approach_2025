import numpy as np


def _mk_samples():
    return [("/fake/video.mp4", 0, 0)]


def _install_frame_cache_stubs(monkeypatch, arr):
    import vkb.cache as vcache
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (arr, {"n": arr.shape[0]}))


def test_train_vs_val_rot360_diff(monkeypatch):
    from vkb import finetune as ft
    # deterministic frame: diagonal gradient
    base = np.arange(32, dtype=np.uint8)
    # make gradient along X so horizontal flip changes content
    fr = np.stack([np.tile(base[None, :], (32, 1))]*3, axis=2)
    _install_frame_cache_stubs(monkeypatch, np.stack([fr], axis=0))

    # stub rotate and zoom to deterministic transforms that change pixels
    monkeypatch.setattr(ft, '_rotate_square', lambda img, ang: img[:, ::-1, :])  # horizontal flip
    monkeypatch.setattr(ft, '_zoom_out', lambda img, out_size, smin, smax: img)  # no-op zoom for simplicity

    ds_tr = ft.FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='train', aug='rot360')
    ds_va = ft.FrameDataset(_mk_samples(), img_size=32, data_root='.', mode='val', aug='rot360')
    x_tr, _ = ds_tr[0]
    x_va, _ = ds_va[0]
    assert (x_tr != x_va).any(), 'rot360 should alter training sample vs validation'


def test_train_vs_val_brightness_diff(monkeypatch):
    from vkb import finetune as ft
    base = np.arange(16, dtype=np.uint8)
    fr = np.stack([np.tile(base[:, None], (1, 16))]*3, axis=2)
    _install_frame_cache_stubs(monkeypatch, np.stack([fr], axis=0))

    # deterministic brightness factor
    import random
    monkeypatch.setattr(random, 'uniform', lambda a, b: 0.5)

    ds_tr = ft.FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='train', aug='none', brightness=0.5)
    ds_va = ft.FrameDataset(_mk_samples(), img_size=16, data_root='.', mode='val', aug='none', brightness=0.5)
    x_tr, _ = ds_tr[0]
    x_va, _ = ds_va[0]
    assert (x_tr != x_va).any(), 'brightness should alter training sample vs validation'


def test_train_vs_val_noise_removed(monkeypatch):
    from vkb import finetune as ft
    fr = np.zeros((8, 8, 3), dtype=np.uint8)
    _install_frame_cache_stubs(monkeypatch, np.stack([fr], axis=0))

    # Noise augmentation was removed; passing noise_std should have no effect
    # Minimal cv2 stub for resize/cvtColor used on val path
    import sys
    class _CV:
        def resize(self, img, sz): return img
        def cvtColor(self, img, code): return img
        COLOR_BGR2RGB = 0
    monkeypatch.setitem(sys.modules, 'cv2', _CV())
    ds_tr = ft.FrameDataset(_mk_samples(), img_size=8, data_root='.', mode='train', aug='none', noise_std=0.3)
    ds_va = ft.FrameDataset(_mk_samples(), img_size=8, data_root='.', mode='val', aug='none', noise_std=0.3)
    x_tr, _ = ds_tr[0]
    x_va, _ = ds_va[0]
    assert (x_tr == x_va).all(), 'noise aug removed: train equals val when only noise_std differed'


def test_train_vs_val_warp_diff(monkeypatch):
    from vkb import finetune as ft
    base = np.arange(10, dtype=np.uint8)
    fr = np.stack([np.tile(base[:, None], (1, 10))]*3, axis=2)
    _install_frame_cache_stubs(monkeypatch, np.stack([fr], axis=0))

    # stub warp to a simple shift-right by 1 with zero fill
    def _fake_warp(img, _frac):
        out = np.zeros_like(img)
        out[:, 1:, :] = img[:, :-1, :]
        return out
    monkeypatch.setattr(ft, '_perspective_warp_inward', _fake_warp)

    ds_tr = ft.FrameDataset(_mk_samples(), img_size=10, data_root='.', mode='train', aug='none', warp=0.2)
    ds_va = ft.FrameDataset(_mk_samples(), img_size=10, data_root='.', mode='val', aug='none', warp=0.2)
    x_tr, _ = ds_tr[0]
    x_va, _ = ds_va[0]
    assert (x_tr != x_va).any(), 'warp should alter training sample vs validation'
