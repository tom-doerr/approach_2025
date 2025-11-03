import sys, unittest


class TestHPOXGBFinalFit(unittest.TestCase):
    def test_final_fit_uses_best_params(self):
        import numpy as np
        import train_frames as tf

        # Dataset: two short videos across two labels
        import os, tempfile
        tmp = tempfile.TemporaryDirectory()
        cwd = os.getcwd(); os.chdir(tmp.name)
        os.makedirs('data/A', exist_ok=True)
        os.makedirs('data/B', exist_ok=True)
        # Use data-rooted paths so cache stays under tmp
        vids = [("data/A/a1.mp4", "A"), ("data/B/b1.mp4", "B"), ("data/B/b2.mp4", "B")]  # ensure ≥2 per class for tail split
        orig_list = tf.list_videos
        tf.list_videos = lambda _: vids

        class FakeCap:
            def __init__(self, path):
                if '/A/' in path:
                    self.frames = [np.full((2,2,3), 0, dtype=np.uint8) for _ in range(6)]
                else:
                    self.frames = [np.full((2,2,3), 255, dtype=np.uint8) for _ in range(6)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass

        class FakeCv: VideoCapture = FakeCap
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        # Embedder: simple mean feature
        orig_create = tf.create_embedder
        tf.create_embedder = lambda *a, **k: (lambda fr: np.array([fr.mean()/255.0], dtype=np.float32))

        # HPO: force best params
        chosen = {
            'max_depth': 5, 'n_estimators': 123, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_lambda': 0.1,
            'tree_method': 'hist', 'n_jobs': 0,
        }
        orig_hpo_xgb = tf._hpo_xgb
        tf._hpo_xgb = lambda *a, **k: (chosen, 1.0, [(chosen, 1.0)])

        captured = {}
        orig_save = tf.save_model
        def fake_save(obj, *a, **k):
            captured['bundle'] = obj
            return 'models/dummy.pkl'
        tf.save_model = fake_save

        from types import SimpleNamespace
        args = SimpleNamespace(data='data', embed_model='mobilenetv3_small_100',
                               eval_split=0.2, eval_mode='tail', clf='xgb', alpha=1.0, hpo_alpha=0, hpo_xgb=1)
        try:
            tf.train(args)
            bundle = captured['bundle']
            clf = bundle['clf']
            params = clf.get_params()
            for k, v in chosen.items():
                self.assertEqual(params[k], v)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass
            tmp.cleanup()
            tf.list_videos = orig_list
            tf.create_embedder = orig_create
            tf._hpo_xgb = orig_hpo_xgb
            tf.save_model = orig_save
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']


if __name__ == '__main__':
    unittest.main()
