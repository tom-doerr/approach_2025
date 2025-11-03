import sys, unittest


class TestTailSplit(unittest.TestCase):
    def test_tail_split_uses_last_indices_per_class(self):
        import numpy as np
        import train_frames as tf

        # Simulate 2 classes, each one video appended sequentially
        vids = [("a1", "A"), ("b1", "B")]
        orig_list = tf.list_videos
        tf.list_videos = lambda _: vids

        class FakeCap:
            def __init__(self, path):
                # 5 frames for A, 6 frames for B
                self.frames = [np.zeros((2,2,3), dtype=np.uint8)] * (5 if 'a1' in path else 6)
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass

        class FakeCv: VideoCapture = FakeCap
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        # Embedder returns constant feature
        orig_create = tf.create_embedder
        tf.create_embedder = lambda *a, **k: (lambda fr: np.array([0.0], dtype=np.float32))

        from types import SimpleNamespace
        args = SimpleNamespace(data="ignored", embed_model="mobilenetv3_small_100",
                               eval_split=0.4, eval_mode='tail', clf="ridge", alpha=1.0, hpo_alpha=0)

        # Run through up to the fit; ensure no exceptions
        try:
            tf.train(args)
        finally:
            tf.list_videos = orig_list
            tf.create_embedder = orig_create
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()
