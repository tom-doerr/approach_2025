import io, os, sys, tempfile, unittest


class TestProgressPrinting(unittest.TestCase):
    def test_progress_lines_present(self):
        import numpy as np
        import train_frames as tf

        tmp = tempfile.mkdtemp()
        vids = [("vid_lab1", "lab1"), ("vid_lab2", "lab2")]
        orig_list = tf.list_videos
        tf.list_videos = lambda _: vids

        class FakeCap:
            def __init__(self, path):
                self.frames = [np.zeros((4,4,3), dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass

        class FakeCv:
            VideoCapture = FakeCap

        old_cv = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv()

        orig_create = tf.create_embedder
        tf.create_embedder = lambda *a, **k: (lambda fr: np.array([0.0], dtype=np.float32))

        from types import SimpleNamespace
        args = SimpleNamespace(data="ignored", embed_model="mobilenetv3_small_100",
                               eval_split=0.0, clf="ridge", alpha=1.0, hpo_alpha=0)
        buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
        try:
            tf.train(args)
        finally:
            sys.stdout = old_stdout
            tf.list_videos = orig_list
            tf.create_embedder = orig_create
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']

        out = buf.getvalue()
        self.assertIn("Embedding 1/2:", out)
        self.assertIn("Training ridge classifier", out)
        self.assertIn("Feature dim:", out)
        self.assertTrue(("FPS" in out) or ("fps" in out) or ("Embedding Speed" in out))
        self.assertIn("alpha", out)


if __name__ == "__main__":
    unittest.main()
