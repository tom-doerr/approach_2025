import io, os, sys, tempfile, unittest


class TestCacheEmbeddings(unittest.TestCase):
    def run_once(self, tmp, expect_hit: bool):
        import numpy as np
        import train_frames as tf
        # one video in one label
        vids = [(os.path.join('data','lab','a.mp4'), 'lab')]
        orig_list = tf.list_videos
        tf.list_videos = lambda root: vids

        class FakeCap:
            def __init__(self, path):
                self.frames = [np.zeros((2,2,3), dtype=np.uint8) for _ in range(3)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass

        class FakeCv: VideoCapture = FakeCap
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        orig_create = tf.create_embedder
        tf.create_embedder = lambda *a, **k: (lambda fr: np.array([0.0, 0.0], dtype=np.float32))

        # Ensure data file exists to make relpath work
        os.makedirs(os.path.join(tmp,'data','lab'), exist_ok=True)
        open(os.path.join(tmp,'data','lab','a.mp4'),'wb').close()

        from types import SimpleNamespace
        args = SimpleNamespace(data='data', embed_model='mobilenetv3_small_100',
                               eval_split=0.0, eval_mode='tail', clf='ridge', alpha=1.0, hpo_alpha=0)
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:
            tf.train(args)
        finally:
            sys.stdout = old
            tf.list_videos = orig_list
            tf.create_embedder = orig_create
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
        out = buf.getvalue()
        if expect_hit:
            assert 'cache hit' in out
        else:
            assert 'cache miss' in out

    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd(); os.chdir(tmp)
            try:
                self.run_once(tmp, expect_hit=False)
                self.run_once(tmp, expect_hit=True)
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
