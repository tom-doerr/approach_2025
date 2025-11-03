import os, sys, glob, pickle, shutil, tempfile, unittest
from types import SimpleNamespace


class TestE2ETrain(unittest.TestCase):
    def test_end_to_end_train_with_fakes(self):
        import numpy as np
        import train_frames as tf

        tmp = tempfile.mkdtemp()
        models_dir = os.path.join(tmp, "models")
        os.makedirs(models_dir)

        # Patch list_videos
        paths = [("vid_lab1", "lab1"), ("vid_lab2", "lab2")]
        orig_list_videos = tf.list_videos
        tf.list_videos = lambda _root: paths

        # Fake cv2 module: only VideoCapture is needed for training
        class FakeCap:
            def __init__(self, path):
                val = 0 if "lab1" in path else 255
                self.frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(6)]
            def isOpened(self):
                return True
            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop()
            def release(self):
                pass

        class FakeCv2:
            VideoCapture = FakeCap

        orig_cv2 = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv2()

        # Stub embedder to a simple mean-intensity feature
        orig_create_embedder = tf.create_embedder
        def fake_embedder(_name, **kwargs):
            def embed(frame):
                return np.array([frame.mean() / 255.0], dtype=np.float32)
            return embed
        tf.create_embedder = fake_embedder

        # Redirect model saving to temp models_dir
        orig_save_model = tf.save_model
        from vkb.artifacts import save_model as real_save
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": real_save(obj, name_parts, base_dir=models_dir, ext=ext)

        try:
            args = SimpleNamespace(data="ignored", embed_model="fake", raw_size=32, raw_rgb=False, eval_split=0.2, eval_mode='tail', clf="ridge", alpha=1.0, hpo_alpha=0)
            tf.train(args)
            files = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
            self.assertTrue(files, "no model saved")
            with open(files[-1], "rb") as f:
                bundle = pickle.load(f)
            clf = bundle["clf"]; labels = bundle["labels"]
            self.assertEqual(set(labels), {"lab1", "lab2"})
            # Verify classifier separates synthetic features
            lab1_feat = np.array([[0.0]], dtype=np.float32)
            lab2_feat = np.array([[1.0]], dtype=np.float32)
            p1 = clf.predict(lab1_feat)[0]
            p2 = clf.predict(lab2_feat)[0]
            self.assertNotEqual(int(p1), int(p2))
        finally:
            # restore patches
            tf.list_videos = orig_list_videos
            tf.create_embedder = orig_create_embedder
            tf.save_model = orig_save_model
            if orig_cv2 is not None:
                sys.modules['cv2'] = orig_cv2
            else:
                del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
