from types import SimpleNamespace as NS


def test_dynamic_aug_off_keeps_defaults():
    # Build minimal args and samples
    from vkb.finetune import _make_loaders
    args = NS(data='.', aug='none', brightness=0.12, warp=0.21, sat=0.08, contrast=0.09, wb=0.07, hue=0.06,
              dynamic_aug=False, batch_size=2, workers=0, prefetch=1, persistent_workers=False, sharing_strategy='auto')
    samples = [("vid.mp4", 0, 0)]
    tr_idx, va_idx = [0], []
    dl_tr, dl_va, dl_te, _ = _make_loaders(args, samples, tr_idx, va_idx, n_classes=1, device='cpu')
    ds = getattr(args, '_ds_tr', None)
    assert ds is not None
    assert ds.brightness == 0.12 and ds.warp == 0.21 and ds.sat == 0.08 and ds.contrast == 0.09 and ds.wb == 0.07 and ds.hue == 0.06

