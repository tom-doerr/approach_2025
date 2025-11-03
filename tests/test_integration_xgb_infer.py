import os, sys, io, glob, pickle, shutil, tempfile, unittest
from types import SimpleNamespace


class TestIntegrationXGBInfer(unittest.TestCase):
    def test_train_and_infer_xgb(self):
        import numpy as np
        import train_frames as tf
        import infer_live as inf

        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos = tf.list_videos
        tf.list_videos = lambda _root: vids

        class FakeCap:
            def __init__(self, path):
                val = 0 if path.endswith("A") else 255
                self.frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(6)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        class FakeCv2Train:
            VideoCapture = FakeCap
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2Train()

        orig_create_embedder = tf.create_embedder
        def fake_embedder(_name, **kwargs):
            def embed(frame):
                # 2-D feature: mean and std
                return np.array([frame.mean()/255.0, frame.std()/255.0], dtype=np.float32)
            return embed
        tf.create_embedder = fake_embedder

        orig_save = tf.save_model
        from vkb.artifacts import save_model as real_save
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)

        try:
            # Train xgboost (no HPO to keep fast)
            args = SimpleNamespace(data=".", embed_model="fake_int_xgb", eval_split=0.2, eval_mode='tail', clf="xgb", hpo_xgb=0)
            tf.train(args)
            path = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))[-1]

            # Wire infer to same embedder and headless cv2
            orig_create_embedder_inf = inf.create_embedder
            inf.create_embedder = fake_embedder
            class FakeCapInfer:
                def __init__(self, *_):
                    self.frames = [np.full((8,8,3), 255, dtype=np.uint8) for _ in range(2)]
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

            buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
            old_argv = sys.argv; sys.argv = ["infer_live.py", "--model-path", path, "--frames", "2"]
            try:
                inf.main()
            finally:
                sys.stdout = old_stdout; sys.argv = old_argv
            out = buf.getvalue()
            assert "Loaded" in out and "Model" in out
            assert "xgboost" in out
        finally:
            tf.list_videos = orig_list_videos
            tf.create_embedder = orig_create_embedder
            tf.save_model = orig_save
            inf.create_embedder = orig_create_embedder_inf
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
