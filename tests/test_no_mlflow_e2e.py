import io, os, sys, tempfile, unittest


class _NoMLflowModule:
    def __getattr__(self, name):
        raise AssertionError("mlflow should not be imported when not requested")


class TestNoMLflowE2E(unittest.TestCase):
    def test_ridge_train_does_not_touch_mlflow(self):
        import numpy as np
        import train_frames as tf
        # Tiny timm model stub so embedder can import timm
        import types, torch
        class TinyNet:
            def __init__(self, num_classes=0):
                self.net = torch.nn.Identity()
            def eval(self): return self
            def __call__(self, x):
                # return a flat feature tensor
                B = x.shape[0]
                return torch.zeros((B, 1))
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = types.SimpleNamespace(create_model=lambda *a, **k: TinyNet())

        # Block mlflow import globally for this run
        sys.modules['mlflow'] = _NoMLflowModule()
        # Also set env opt-out just in case
        os.environ['VKB_MLFLOW_DISABLE'] = '1'

        # Fake videos
        vids = [("vid_lab1", "lab1"), ("vid_lab2", "lab2")]
        orig_list = tf.list_videos
        tf.list_videos = lambda _: vids

        class FakeCap:
            def __init__(self, path):
                self.frames = [np.zeros((4,4,3), dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        class CV2:
            VideoCapture = FakeCap
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = CV2()

        # Redirect artifacts to temp
        tmp = tempfile.mkdtemp()
        from vkb.artifacts import save_model as real_save
        orig_save = tf.save_model
        tf.save_model = lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=os.path.join(tmp, 'models'), ext=ext)

        from types import SimpleNamespace
        args = SimpleNamespace(data='ignored', embed_model='mobilenetv3_small_100',
                               eval_split=0.0, eval_mode='tail', clf='ridge', alpha=1.0, hpo_alpha=0)
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:
            tf.train(args)
        finally:
            sys.stdout = old
            tf.list_videos = orig_list
            tf.save_model = orig_save
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: sys.modules.pop('timm', None)
        out = buf.getvalue()
        assert 'MLflow' not in out

    def test_dl_cli_does_not_touch_mlflow(self):
        import numpy as np
        import train_frames as tf

        # Block mlflow imports and disable via env
        sys.modules['mlflow'] = _NoMLflowModule()
        os.environ['VKB_MLFLOW_DISABLE'] = '1'

        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        # Fake videos
        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_vio = vio.list_videos
        vio.list_videos = lambda _root: vids

        class FakeCap:
            def __init__(self, path):
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        class FakeCv2:
            VideoCapture = FakeCap
            def resize(self, img, sz): return img
            def cvtColor(self, img, code): return img
            COLOR_BGR2RGB = 0
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2()

        # Tiny timm model stub
        import types
        class TinyNet:
            def __init__(self, num_classes=2):
                import torch, torch.nn as nn
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
            def to(self, d): return self
            def parameters(self): return self.net.parameters()
            def load_state_dict(self, sd, strict=False): return self
            def state_dict(self): return self.net.state_dict()
            def __call__(self, x): return self.net(x)
            def train(self): return self
            def eval(self): return self
        fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=True, num_classes=2, **_: TinyNet(num_classes))
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm

        # Redirect model save to temp (for finetune path)
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
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2; vio2.list_videos = orig_list_videos_vio
            tf.save_model = orig_save
            import vkb.artifacts as vart2; vart2.save_model = orig_vart_save
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']

        out = buf.getvalue()
        assert 'MLflow' not in out


if __name__ == "__main__":
    unittest.main()
