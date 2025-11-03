import io, os, sys, types


def test_cache_summary_prints_after_epoch_one(monkeypatch):
    import vkb.finetune as ft
    # Stub videos and samples
    vids = [("vidA.mp4","A"),("vidB.mp4","A")]
    monkeypatch.setattr(ft, '_labels_and_videos', lambda root: (vids, ["A","B"], {"A":0, "B":1}))
    monkeypatch.setattr(ft, '_build_samples', lambda vids, lab2i, stride=1, stride_offset=0: [(v, i, 0) for v in ["vidA.mp4","vidB.mp4"] for i in (0,1)])

    # Stub cache: one miss, one hit
    hits = {"vidA.mp4": False, "vidB.mp4": True}
    import vkb.cache as vc
    def ens(_root, vp):
        # h=4, w=5, n=2
        return {"n":2, "h":4, "w":5}, (not hits.get(vp, True))
    def opn(_root, vp):
        import numpy as np
        meta = {"n":2, "h":4, "w":5}
        arr = np.zeros((2,4,5,3), dtype=np.uint8)
        return arr, meta
    monkeypatch.setattr(vc, 'ensure_frames_cached', ens)
    monkeypatch.setattr(vc, 'open_frames_memmap', opn)

    # Stub timm with a tiny parametric model
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self, num_classes=1):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(2, int(num_classes))
        def forward(self, x):
            x = self.pool(self.conv(x)).flatten(1)
            return self.fc(x)
    sys.modules['timm'] = types.SimpleNamespace(create_model=lambda name, pretrained=False, num_classes=1, **kw: Tiny(num_classes))

    # Run one epoch and capture output
    args = ft.FTArgs(data='.', embed_model='stub', eval_split=0.0, eval_mode='tail', epochs=1, batch_size=2, lr=1e-4, wd=1e-4, device='cpu')
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        ft.finetune(args)
    finally:
        sys.stdout = old
        sys.modules.pop('timm', None)
    out = buf.getvalue()
    assert 'Cache: videos=' in out and 'hits=' in out and 'misses=' in out
