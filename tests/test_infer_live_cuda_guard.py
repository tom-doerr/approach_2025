import os, sys, io, pickle, tempfile, shutil, unittest


class TestInferLiveCUDAGuard(unittest.TestCase):
    def test_cuda_requested_without_gpu_raises(self):
        import infer_live as inf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)
        # minimal DL bundle
        bundle = {"clf_name": "finetune", "labels": ["A","B"], "model_name": "mobilenetv3_small_100", "state_dict": {}, "input_size": 8, "normalize": {"mean":[0,0,0],"std":[1,1,1]}}
        path = os.path.join(models_dir, "20250101_000000_finetune.pkl")
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # stub timm, torch(cuda off), and cv2 so we don't touch hardware
        import types
        class TinyNet:
            def __init__(self, num_classes=2): pass
            def load_state_dict(self, *a, **k): pass
            def eval(self): return self
            def to(self, *_): return self
            def __call__(self, x):
                import torch
                return torch.zeros((x.shape[0], 2))
        fake_timm = types.SimpleNamespace(create_model=lambda *_a, **_k: TinyNet())
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm

        import types as _types
        import torch as _torch
        # Force cuda unavailable
        old_cuda = _torch.cuda.is_available
        _torch.cuda.is_available = lambda: False

        class FakeCv:
            def VideoCapture(self, *_):
                class C: 
                    def isOpened(self): return True
                    def read(self): return False, None
                    def release(self): pass
                return C()
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

        # run and expect a RuntimeError
        old_argv = sys.argv
        try:
            sys.argv = ["infer_live.py", "--model-path", path, "--frames", "1", "--device", "cuda"]
            with self.assertRaises(RuntimeError):
                inf.main()
        finally:
            sys.argv = old_argv
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            _torch.cuda.is_available = old_cuda
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

