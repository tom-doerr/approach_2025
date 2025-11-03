import os


def test_models_dir_env_overrides(tmp_path, monkeypatch):
    import vkb.artifacts as art
    monkeypatch.setenv('VKB_MODELS_DIR', str(tmp_path))
    p = art.save_model({'a':1}, ['unit','test'])
    assert str(tmp_path) in p
    latest = art.latest_model()
    assert latest == p


def test_visdom_disable_env(monkeypatch):
    from vkb.finetune import _setup_visdom
    monkeypatch.setenv('VKB_VISDOM_DISABLE', '1')
    class A: visdom_aug=0; visdom_metrics=False
    v = _setup_visdom(A(), type('C', (), {'print': lambda *a, **k: None})())
    assert v is None


def test_mlflow_disable_env(monkeypatch):
    import train_frames as tf
    from types import SimpleNamespace as NS
    # Ensure mlflow disabled even if flag set
    monkeypatch.setenv('VKB_MLFLOW_DISABLE', '1')
    # Minimal stubs to avoid heavy deps; one tiny video, 1 frame
    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)
    import numpy as np, sys
    monkeypatch.setattr(tf, 'create_embedder', lambda name: (lambda fr: np.array([1.0, 2.0], dtype=np.float32)))
    class FakeCap:
        def __init__(self, path): self.frames = [np.zeros((4,4,3), dtype=np.uint8)]
        def isOpened(self): return True
        def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
        def release(self): pass
    class CV2: VideoCapture=FakeCap
    monkeypatch.setitem(sys.modules, 'cv2', CV2())
    # Avoid writing to real models dir
    import vkb.artifacts as vart
    from vkb.artifacts import save_model as real_save
    import tempfile
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(vart, 'save_model', lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    args = NS(data='.', embed_model='fake', clf='ridge', hpo_xgb=0, eval_split=0.0,
              mlflow=True, mlflow_uri=None, mlflow_exp='vkb', mlflow_run_name=None,
              alpha=1.0, hpo_alpha=0, C=1.0, hpo_logreg=0)
    # Should not raise even without mlflow installed
    tf.train(args)
