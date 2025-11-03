import os, sys, io, glob, shutil, tempfile, unittest


class TestE2ECLIRidgeLatest(unittest.TestCase):
    def test_train_cli_and_infer_latest(self):
        import numpy as np
        import train_frames as tf
        import infer_live as inf

        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        # Patch list_videos for both modules
        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_vio = vio.list_videos
        vio.list_videos = lambda _root: vids

        # Headless cv2 for train and infer
        class FakeCapTrain:
            def __init__(self, path):
                val = 0 if path.endswith("A") else 255
                self.frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(4)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
        class FakeCv2Train:
            VideoCapture = FakeCapTrain
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2Train()

        # Minimal embedder: 1-D mean feature; patch in both places
        orig_create_embedder_tf = tf.create_embedder
        def fake_embedder(_name, **kwargs):
            def embed(fr):
                return np.array([fr.mean()/255.0], dtype=np.float32)
            return embed
        tf.create_embedder = fake_embedder

        # Redirect model save to temp
        orig_save = tf.save_model
        from vkb.artifacts import save_model as real_save, latest_model
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)

        # Train via CLI entry (main) with unique embed_model to avoid cache cross-talk
        old_argv = sys.argv
        try:
            sys.argv = [
                "train_frames.py",
                "--data", ".",
                "--embed-model", "fake_cli_ridge",
                "--clf", "ridge",
                "--alpha", "1.0",
                "--hpo-alpha", "0",
                "--eval-split", "0.0",
                "--allow-test-labels",
            ]
            tf.main()
            # Resolve latest
            p = latest_model(base_dir=models_dir)
            assert os.path.exists(p)

            # Now run infer_live without model-path; patch its embedder + cv2
            orig_create_embedder_inf = inf.create_embedder
            inf.create_embedder = fake_embedder
            class FakeCapInfer:
                def __init__(self, *_):
                    self.frames = [np.full((8,8,3), 0, dtype=np.uint8) for _ in range(2)]
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
            try:
                sys.argv = ["infer_live.py", "--frames", "2"]
                inf.main()
            finally:
                sys.stdout = old_stdout
                inf.create_embedder = orig_create_embedder_inf
        finally:
            sys.argv = old_argv
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2; vio2.list_videos = orig_list_videos_vio
            tf.create_embedder = orig_create_embedder_tf
            tf.save_model = orig_save
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
