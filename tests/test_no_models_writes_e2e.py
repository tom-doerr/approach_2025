import io, os, sys, tempfile, unittest


def _snapshot_models_dir():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mdir = os.path.join(root, "models")
    try:
        names = sorted(os.listdir(mdir))
    except FileNotFoundError:
        names = []
    stats = {n: (os.stat(os.path.join(mdir, n)).st_mtime_ns,
                 os.stat(os.path.join(mdir, n)).st_size) for n in names if os.path.exists(os.path.join(mdir, n))}
    return mdir, names, stats


class TestNoModelsWritesE2E(unittest.TestCase):
    def test_ridge_train_does_not_write_repo_models(self):
        import numpy as np
        import train_frames as tf
        # snapshot before
        mdir, names0, stats0 = _snapshot_models_dir()
        # isolate cache/models
        with tempfile.TemporaryDirectory() as tmp:
            # monkeypatch save path to temp
            orig_save = tf.save_model
            from vkb.artifacts import save_model as real_save
            tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=os.path.join(tmp, "models"), ext=ext)
            # tiny stubs
            vids = [("vid_lab1", "lab1"), ("vid_lab2", "lab2")]
            orig_list = tf.list_videos; tf.list_videos = lambda _: vids
            class FakeCap:
                def __init__(self, path):
                    self.frames = [np.zeros((4,4,3), dtype=np.uint8) for _ in range(2)]
                def isOpened(self): return True
                def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
                def release(self): pass
            class FakeCv: VideoCapture = FakeCap
            old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
            orig_create = tf.create_embedder; tf.create_embedder = lambda *a, **k: (lambda fr: np.array([0.0], dtype=np.float32))
            from types import SimpleNamespace
            args = SimpleNamespace(data="ignored", embed_model="mobilenetv3_small_100",
                                   eval_split=0.0, clf="ridge", alpha=1.0, hpo_alpha=0)
            buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
            try:
                tf.train(args)
            finally:
                sys.stdout = old_stdout
                tf.list_videos = orig_list; tf.save_model = orig_save; tf.create_embedder = orig_create
                if old_cv is not None: sys.modules['cv2'] = old_cv
                else: del sys.modules['cv2']
        # snapshot after
        _, names1, stats1 = _snapshot_models_dir()
        assert names1 == names0 and stats1 == stats0, f"models/ changed: before={names0} after={names1}"

    def test_dl_cli_does_not_write_repo_models(self):
        import numpy as np
        import train_frames as tf
        os.environ['VKB_VISDOM_DISABLE'] = '1'
        mdir, names0, stats0 = _snapshot_models_dir()
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)
            vids = [("vid_A", "A"), ("vid_B", "B")]
            orig_list_videos_tf = tf.list_videos; tf.list_videos = lambda _root: vids
            import vkb.io as vio
            orig_list_videos_vio = vio.list_videos; vio.list_videos = lambda _root: vids
            class FakeCap:
                def __init__(self, path):
                    val = 32 if path.endswith("A") else 224
                    self.frames = [np.full((16,16,3), val, dtype=np.uint8) for _ in range(4)]
                def isOpened(self): return True
                def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
                def release(self): pass
            class FakeCv2:
                VideoCapture = FakeCap
                def resize(self, img, sz): return img
                def cvtColor(self, img, code): return img
                COLOR_BGR2RGB = 0
            old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2()
            import types
            class TinyNet:
                def __init__(self, num_classes=2):
                    import torch, torch.nn as nn
                    self.net = nn.Sequential(nn.Flatten(), nn.Linear(16*16*3, num_classes))
                def to(self, d): return self
                def parameters(self): return self.net.parameters()
                def load_state_dict(self, sd, strict=False): return self
                def state_dict(self): return self.net.state_dict()
                def __call__(self, x): return self.net(x)
                def train(self): return self
                def eval(self): return self
            fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=True, num_classes=2, **_: TinyNet(num_classes))
            old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm
            orig_save = tf.save_model
            from vkb.artifacts import save_model as real_save
            tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)
            import vkb.artifacts as vart
            orig_vart_save = vart.save_model
            vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_vart_save(obj, name_parts, base_dir=models_dir, ext=ext)
            old_argv = sys.argv; buf = io.StringIO(); old_stdout = sys.stdout
            try:
                sys.argv = [
                    "train_frames.py",
                    "--data", ".",
                    "--clf", "dl",
                    "--embed-model", "mobilenetv3_small_100",
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--device", "cpu",
                    "--eval-split", "0.2",
                    "--allow-test-labels",
                    "--aug", "none",
                    "--warp", "0",
                    "--brightness", "0.0",
                ]
                sys.stdout = buf
                tf.main()
            finally:
                sys.stdout = old_stdout; sys.argv = old_argv
                tf.list_videos = orig_list_videos_tf; tf.save_model = orig_save
                import vkb.artifacts as vart2; vart2.save_model = orig_vart_save
                import vkb.io as vio2; vio2.list_videos = orig_list_videos_vio
                if old_cv2 is not None: sys.modules['cv2'] = old_cv2
                else: del sys.modules['cv2']
                if old_timm is not None: sys.modules['timm'] = old_timm
                else: del sys.modules['timm']
        _, names1, stats1 = _snapshot_models_dir()
        assert names1 == names0 and stats1 == stats0, f"models/ changed: before={names0} after={names1}"


if __name__ == "__main__":
    unittest.main()
