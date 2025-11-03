import os, sys, io, glob, pickle, shutil, tempfile, unittest


class TestE2ECLIDLLatest(unittest.TestCase):
    def test_train_cli_dl_and_infer(self):
        import train_frames as tf
        import infer_live as inf

        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        # Fake videos and frames
        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_vio = vio.list_videos
        vio.list_videos = lambda _root: vids

        import numpy as np
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
            def load_state_dict(self, sd, strict=False):
                return self
            def state_dict(self): return self.net.state_dict()
            def __call__(self, x): return self.net(x)
            def train(self): return self
            def eval(self): return self
        fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=True, num_classes=2: TinyNet(num_classes))
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm

        # Redirect model save to temp
        orig_save = tf.save_model
        from vkb.artifacts import save_model as real_save, latest_model
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)
        import vkb.artifacts as vart
        orig_save_vart = vart.save_model
        vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_save_vart(obj, name_parts, base_dir=models_dir, ext=ext)

        # Train DL via CLI
        old_argv = sys.argv
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
            tf.main()
            p = latest_model(base_dir=models_dir)
            assert os.path.exists(p)

            # Now run infer_live with DL path
            class FakeCapInfer:
                def __init__(self, *_):
                    self.frames = [np.zeros((16,16,3), dtype=np.uint8) for _ in range(2)]
                def isOpened(self): return True
                def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
                def release(self): pass
            class FakeCv2Infer:
                VideoCapture = FakeCapInfer
                def waitKey(self, *_): return ord('q')
                def imshow(self, *_): pass
                def destroyAllWindows(self): pass
                def putText(self, *a, **k): pass
                FONT_HERSHEY_SIMPLEX = 0
                LINE_AA = 0
                def cvtColor(self, img, code): return img
                def resize(self, img, sz): return img
                COLOR_BGR2RGB = 0
            sys.modules['cv2'] = FakeCv2Infer()

            # stub timm for infer too
            import infer_live as inf2
            old_timm_inf = sys.modules.get('timm')
            sys.modules['timm'] = fake_timm
            buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
            try:
                sys.argv = ["infer_live.py", "--frames", "2", "--device", "cpu"]
                inf2.main()
            finally:
                sys.stdout = old_stdout
                sys.modules['timm'] = old_timm_inf if old_timm_inf is not None else sys.modules.pop('timm', None)
        finally:
            sys.argv = old_argv
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2; vio2.list_videos = orig_list_videos_vio
            tf.save_model = orig_save
            import vkb.artifacts as vart2
            vart2.save_model = orig_save_vart
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
