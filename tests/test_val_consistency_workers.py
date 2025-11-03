import os, sys, tempfile, types, json


def _run_dl(monkeypatch, workers: int):
    import train_frames as tf
    import vkb.io as vio

    # Deterministic seeds
    import random as _r, numpy as _np
    try:
        import torch as _t
        _t.manual_seed(0)
    except Exception:
        pass
    _r.seed(0); _np.random.seed(0)

    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, "list_videos", lambda root: vids)
    monkeypatch.setattr(vio, "list_videos", lambda root: vids)

    # cv2 stub (fixed values, no randomness)
    class FakeCv2:
        COLOR_BGR2RGB = 0
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(4)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        def resize(self, img, sz): return img
        def cvtColor(self, img, code): return img
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())

    # timm stub: tiny linear model; torch seed above makes weights deterministic
    import torch, torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
        def forward(self, x): return self.net(x)
    fake_timm = types.SimpleNamespace(create_model=lambda name, **kw: TinyNet(kw.get('num_classes', 2)))
    monkeypatch.setitem(sys.modules, 'timm', fake_timm)

    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    import vkb.artifacts as vart
    monkeypatch.setattr(vart, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))

    from types import SimpleNamespace
    args = SimpleNamespace(
        data='.', embed_model='mobilenetv3_small_100', backbone='mobilenetv3_small_100',
        clf='dl', device='cpu', epochs=2, batch_size=2, eval_split=0.5, eval_mode='tail', allow_test_labels=True,
        aug='none', noise_std=0.0, brightness=0.0, warp=0.0, rot_deg=0.0, shift=0.0, sat=0.0, contrast=0.0, wb=0.0, hue=0.0,
        stride=1, class_weights='none', lr=1e-3, wd=0.0, drop_path=0.0, dropout=0.0,
        workers=workers, prefetch=1, persistent_workers=False, sharing_strategy='auto',
        rich_progress=False, visdom_aug=0, visdom_metrics=False, visdom_env='vkb-aug', visdom_port=8097)
    p = tf.train(args)
    meta = p + ".meta.json"
    with open(meta, 'r') as f:
        m = json.load(f)
    return float(m.get('val_acc') or 0.0)


def test_val_acc_consistent_workers(monkeypatch):
    v0 = _run_dl(monkeypatch, workers=0)
    v2 = _run_dl(monkeypatch, workers=2)
    # Exact equality expected in this deterministic setup
    assert abs(v0 - v2) < 1e-6

