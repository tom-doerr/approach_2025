import os, sys, io, pickle, tempfile, shutil, unittest


class TestInferLiveDL(unittest.TestCase):
    def test_dl_path_runs_n_frames(self):
        import infer_live as inf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)
        # fake bundle
        bundle = {
            "clf_name": "finetune",
            "labels": ["A","B"],
            "model_name": "mobilenetv3_small_100",
            "state_dict": {},
            "input_size": 8,
            "normalize": {"mean":[0.0,0.0,0.0], "std":[1.0,1.0,1.0]},
        }
        path = os.path.join(models_dir, "20250101_000000_finetune.pkl")
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # stub timm and torch model
        import types
        class TinyNet:
            def __init__(self, num_classes=2):
                import torch, torch.nn as nn
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
            def load_state_dict(self, sd, strict=False):
                return
            def eval(self):
                return self
            def to(self, dev):
                return self
            def __call__(self, x):
                import torch
                return torch.zeros((x.shape[0], 2))
        fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=False, num_classes=2: TinyNet(num_classes))
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm

        # stub cv2: VideoCapture yields a few frames; imshow/waitKey no-ops
        import numpy as np
        class FakeCap:
            def __init__(self, *_):
                self.frames = [np.zeros((8,8,3), dtype=np.uint8) for _ in range(3)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        class FakeCv:
            VideoCapture = FakeCap
            def waitKey(self, *_): return ord('q')
            def imshow(self, *_): pass
            def destroyAllWindows(self): pass
            def putText(self, *a, **k): pass
            FONT_HERSHEY_SIMPLEX = 0
            LINE_AA = 0
            def cvtColor(self, img, code): return img
            def resize(self, img, sz): return img
            COLOR_BGR2RGB = 0
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        # run with limited frames
        old_argv = sys.argv
        buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
        try:
            sys.argv = ["infer_live.py", "--model-path", path, "--frames", "2", "--device", "cpu"]
            inf.main()
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

