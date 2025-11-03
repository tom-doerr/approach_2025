import os, sys, io, json, tempfile, shutil, unittest
try:
    import optuna  # noqa: F401
except Exception:  # skip test when optuna is not installed
    import pytest
    pytest.skip("optuna not installed", allow_module_level=True)


class TestOptunaFinetuneCLI(unittest.TestCase):
    def test_optuna_single_trial(self):
        import optuna  # require real optuna

        # Fakes: videos, cv2, timm
        vids = [("vid_A", "A"), ("vid_B", "B")]
        import vkb.io as vio
        orig_list_videos = vio.list_videos
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
            def copyMakeBorder(self, img, *a, **k): return img
            COLOR_BGR2RGB = 0
            BORDER_REFLECT_101 = 0
            def getPerspectiveTransform(self, *a, **k):
                import numpy as _np
                return _np.eye(3, dtype=_np.float32)
            def warpPerspective(self, img, *a, **k): return img
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

        # Redirect artifacts to temp
        from vkb.artifacts import save_model as real_save
        import vkb.artifacts as vart
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)
        orig_save = vart.save_model
        vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)

        # Run CLI main with 1 trial
        import optuna_finetune as cli
        old_argv = sys.argv
        buf = io.StringIO(); old_stdout = sys.stdout
        try:
            sys.argv = [
                "optuna_finetune.py",
                "--data", ".",
                "--trials", "1",
                "--epochs", "1",
                "--batch-size", "2",
                "--device", "cpu",
            ]
            sys.stdout = buf
            cli.main()
            out = buf.getvalue()
            assert "best_value=" in out and "best_params=" in out
            # Verify a sidecar with val_acc exists
            metas = [p for p in os.listdir(models_dir) if p.endswith('.pkl.meta.json')]
            assert metas, "no sidecar written"
            with open(os.path.join(models_dir, metas[-1])) as f:
                meta = json.load(f)
            assert "val_acc" in meta
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv
            vio.list_videos = orig_list_videos
            vart.save_model = orig_save
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
