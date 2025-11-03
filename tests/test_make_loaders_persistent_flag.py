import types


def _dummy_ds(n=8):
    class DS:
        def __init__(self, *a, **k):
            self.samples = list(range(n))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            import torch
            # 3x224x224 zeros + label 0
            return torch.zeros(3,224,224), torch.tensor(0)
    return DS


def test_make_loaders_respects_no_persistent_workers(monkeypatch):
    from vkb import finetune as ft
    # Swap out FrameDataset with a tiny dummy to avoid I/O
    monkeypatch.setattr(ft, 'FrameDataset', _dummy_ds())

    args = types.SimpleNamespace(
        data='data_small', aug='none', brightness=0.0, warp=0.0,
        sat=0.0, contrast=0.0, wb=0.0, hue=0.0, rot_deg=0.0,
        workers=2, prefetch=1, persistent_workers=True, batch_size=4,
        class_weights='none'
    )
    samples = [("v.mp4", i, 0) for i in range(20)]
    tr_idx = list(range(15)); va_idx = list(range(15,20))
    dl_tr, dl_va, _dl_te, cfg = ft._make_loaders(args, samples, tr_idx, va_idx, 1, device='cpu')
    assert cfg[3] is True  # persistent on by default

    # Now disable via explicit negation
    args.no_persistent_workers = True
    dl_tr, dl_va, _dl_te, cfg = ft._make_loaders(args, samples, tr_idx, va_idx, 1, device='cpu')
    assert cfg[3] is False
