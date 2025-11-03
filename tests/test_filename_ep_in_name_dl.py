import os, sys, tempfile, types


def test_dl_model_filename_includes_best_epoch(monkeypatch):
    import train_frames as tf
    import vkb.io as vio

    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, "list_videos", lambda root: vids)
    monkeypatch.setattr(vio, "list_videos", lambda root: vids)

    class FakeCv2:
        COLOR_BGR2RGB = 0
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        def resize(self, img, sz): return img
        def cvtColor(self, img, code): return img
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())

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
    args = SimpleNamespace(data='.', embed_model='mobilenetv3_small_100', backbone='mobilenetv3_small_100', clf='dl', device='cpu', epochs=1, batch_size=2, eval_split=0.5, eval_mode='tail', allow_test_labels=True, aug='none', noise_std=0.0, brightness=0.0, warp=0.0, stride=1, class_weights='none', lr=1e-4, wd=1e-4, drop_path=0.0, dropout=0.0, workers=0, prefetch=1, persistent_workers=False, sharing_strategy='auto', rich_progress=False, visdom_aug=0, visdom_metrics=False, visdom_env='vkb-aug', visdom_port=8097)
    tf.train(args)

    files = [f for f in os.listdir(tmp) if f.endswith('.pkl')]
    assert files, 'expected a saved model file'
    # best epoch should be 1 → ep01
    assert any('_ep01' in f for f in files), 'expected filename to include _epXX'

