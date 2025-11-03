import os, sys, tempfile, types


def test_sidecar_includes_testacc_classic(monkeypatch):
    import train_frames as tf
    vids = [("vid_A", "A"), ("vid_B", "B")]
    # Make both train/val/test possible by giving 10 frames per class
    class FakeCv2:
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(10)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)
    import vkb.io as vio
    monkeypatch.setattr(vio, 'list_videos', lambda root: vids)
    # Embedder stub
    import numpy as np
    monkeypatch.setattr(tf, 'create_embedder', lambda name: (lambda fr: np.array([1.0, 2.0], dtype=np.float32)))
    # Redirect models dir
    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    # Train
    from types import SimpleNamespace
    args = SimpleNamespace(data='.', embed_model='fake', clf='ridge', alpha=1.0, hpo_alpha=0, eval_split=0.1, test_split=0.1, eval_mode='tail')
    tf.train(args)
    # Check sidecar has test_acc
    metas = [f for f in os.listdir(tmp) if f.endswith('.pkl.meta.json')]
    assert metas, 'expected sidecar'
    import json
    meta = json.load(open(os.path.join(tmp, metas[0])))
    assert 'test_acc' in meta


def test_sidecar_includes_testacc_dl(monkeypatch):
    import train_frames as tf
    import vkb.io as vio
    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)
    monkeypatch.setattr(vio, 'list_videos', lambda root: vids)
    # cv2 stub (10 frames per class to allow splits)
    class FakeCv2:
        COLOR_BGR2RGB = 0
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(10)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        def resize(self, img, sz): return img
        def cvtColor(self, img, code): return img
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())
    # timm stub: tiny linear model
    import torch, torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
        def forward(self, x): return self.net(x)
    fake_timm = types.SimpleNamespace(create_model=lambda name, **kw: TinyNet(kw.get('num_classes', 2)))
    monkeypatch.setitem(sys.modules, 'timm', fake_timm)
    # Redirect models dir for both classic and DL saver
    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    import vkb.artifacts as vart
    monkeypatch.setattr(vart, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    # Train DL 1 epoch CPU
    from types import SimpleNamespace
    args = SimpleNamespace(data='.', embed_model='mobilenetv3_small_100', backbone='mobilenetv3_small_100', clf='dl', device='cpu', epochs=1, batch_size=2,
                           eval_split=0.1, test_split=0.1, eval_mode='tail', allow_test_labels=True, aug='none', brightness=0.0, warp=0.0,
                           class_weights='none', lr=1e-4, wd=1e-4, drop_path=0.0, dropout=0.0, workers=0, prefetch=1, persistent_workers=False,
                           sharing_strategy='auto', rich_progress=False, visdom_aug=0, visdom_metrics=False, visdom_env='vkb-aug', visdom_port=8097)
    tf.train(args)
    metas = [f for f in os.listdir(tmp) if f.endswith('.pkl.meta.json')]
    assert metas, 'expected sidecar'
    import json
    meta = json.load(open(os.path.join(tmp, metas[0])))
    assert 'test_acc' in meta
