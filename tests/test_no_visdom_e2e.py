import io, os, sys, tempfile, unittest


class _NoVisdomModule:
    def __getattr__(self, name):
        raise AssertionError("visdom should not be imported when VKB_VISDOM_DISABLE=1")


class TestNoVisdomE2E(unittest.TestCase):
    def test_dl_cli_does_not_touch_visdom_when_disabled(self):
        import numpy as np
        import train_frames as tf

        # Disable visdom globally for this run
        os.environ['VKB_VISDOM_DISABLE'] = '1'
        # Guard: fail if code tries to import from visdom
        sys.modules['visdom'] = _NoVisdomModule()

        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        # Fake videos and frames
        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_vio = vio.list_videos
        vio.list_videos = lambda _root: vids

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

        # Tiny timm model stub
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

        # Redirect model save to temp (both train_frames and vkb.finetune)
        orig_save = tf.save_model
        from vkb.artifacts import save_model as real_save
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)
        import vkb.artifacts as vart
        orig_vart_save = vart.save_model
        vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_vart_save(obj, name_parts, base_dir=models_dir, ext=ext)

        # Train DL via CLI with visdom disabled
        old_argv = sys.argv
        buf = io.StringIO(); old_stdout = sys.stdout
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
            sys.stdout = old_stdout
            sys.argv = old_argv
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2; vio2.list_videos = orig_list_videos_vio
            tf.save_model = orig_save
            import vkb.artifacts as vart2; vart2.save_model = orig_vart_save
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']

        out = buf.getvalue()
        assert "Visdom:" not in out


if __name__ == "__main__":
    unittest.main()
