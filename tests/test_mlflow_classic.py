import os, sys, tempfile


class _FakeMLF:
    def __init__(self):
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.uri = None
        self.exp = None
    def set_tracking_uri(self, uri): self.uri = uri
    def set_experiment(self, name): self.exp = name
    def start_run(self, run_name=None):
        self.run_name = run_name; return self
    def end_run(self): pass
    def log_params(self, d): self.params.append(dict(d))
    def log_param(self, k, v): self.params.append({k: v})
    def log_metric(self, k, v, step=None): self.metrics.append((k, float(v), int(step or 0)))
    def log_artifact(self, path): self.artifacts.append(os.path.basename(path))


def test_mlflow_logging_classic(monkeypatch):
    import types
    import numpy as np
    import train_frames as tf

    # Stub videos and embedder
    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)
    # Patch the imported symbol inside train_frames
    monkeypatch.setattr(tf, 'create_embedder', lambda name: (lambda fr: np.array([1.0, 2.0], dtype=np.float32)))

    # Minimal cv2 stub to feed frames
    class FakeCap:
        def __init__(self, path):
            self.frames = [np.zeros((8,8,3), dtype=np.uint8) for _ in range(2)]
        def isOpened(self): return True
        def read(self):
            return (True, self.frames.pop()) if self.frames else (False, None)
        def release(self): pass
    class CV2:
        VideoCapture = FakeCap
    monkeypatch.setitem(sys.modules, 'cv2', CV2())

    # Temporary models dir
    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, 'save_model', lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))
    import vkb.artifacts as vart
    monkeypatch.setattr(vart, 'save_model', lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))

    fake = _FakeMLF(); monkeypatch.setitem(sys.modules, 'mlflow', fake)
    from types import SimpleNamespace as NS
    args = NS(data='.', embed_model='fake', clf='ridge', alpha=1.0, hpo_alpha=0, hpo_xgb=0, C=1.0, hpo_logreg=0,
              eval_split=0.2, eval_mode='tail', mlflow=True, mlflow_uri=None, mlflow_exp='vkb', mlflow_run_name='test-classic')
    tf.train(args)
    # assertions
    assert fake.params, 'expected mlflow.log_params calls'
    assert any('clf' in d for d in fake.params), 'params should include clf'
    assert fake.artifacts, 'expected logged artifacts'
