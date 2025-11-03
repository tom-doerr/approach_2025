import numpy as np


def test_dspy_uses_cli_defaults_as_start(monkeypatch):
    # Patch cache to avoid real I/O
    import vkb.cache as vcache
    monkeypatch.setattr(vcache, 'ensure_frames_cached', lambda *a, **k: None)
    mm = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(vcache, 'open_frames_memmap', lambda *a, **k: (mm, {"n": 1, "h": 8, "w": 8}))

    import train_frames as tf
    import vkb.finetune as ft

    # Defaults from CLI (we don't pass any overrides): dynamic aug stays off by default
    args = tf.parse_args([])
    args.clf = 'dl'
    args.device = 'cpu'
    # Build minimal sample set
    samples = [("vid.mp4", 0, 0)]
    tr_idx, va_idx = [0], []
    dl_tr, dl_va, dl_te, _ = ft._make_loaders(args, samples, tr_idx, va_idx, n_classes=1, device='cpu')
    ds = getattr(args, '_ds_tr', None)
    assert ds is not None
    # Assert dataset starts with CLI defaults (not zeros)
    assert abs(ds.brightness - args.brightness) < 1e-9
    assert abs(ds.warp - args.warp) < 1e-9
    assert abs(ds.sat - args.sat) < 1e-9
    assert abs(ds.contrast - args.contrast) < 1e-9
    assert abs(ds.wb - args.wb) < 1e-9
    assert abs(ds.hue - args.hue) < 1e-9
    assert abs(ds.rot_deg - args.rot_deg) < 1e-9
