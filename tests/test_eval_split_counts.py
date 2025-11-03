import sys, unittest


class TestEvalSplitCounts(unittest.TestCase):
    def test_raises_clear_error_when_class_underpopulated(self):
        import numpy as np
        import train_frames as tf

        # One frame for A, many for B
        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list = tf.list_videos
        tf.list_videos = lambda _: vids

        class FakeCap:
            def __init__(self, path):
                self.frames = [np.zeros((4,4,3), dtype=np.uint8)] if 'vid_A' in path else [np.zeros((4,4,3), dtype=np.uint8) for _ in range(5)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass

        class FakeCv: VideoCapture = FakeCap
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        orig_create = tf.create_embedder
        tf.create_embedder = lambda *a, **k: (lambda fr: np.array([0.0], dtype=np.float32))

        from types import SimpleNamespace
        args = SimpleNamespace(data="ignored", embed_model="mobilenetv3_small_100",
                               eval_split=0.2, clf="ridge", alpha=1.0, hpo_alpha=0)
        try:
            with self.assertRaises(ValueError) as ctx:
                tf.train(args)
            self.assertIn('>=2 samples per class', str(ctx.exception))
            self.assertIn('A(1)', str(ctx.exception))
        finally:
            tf.list_videos = orig_list
            tf.create_embedder = orig_create
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()
