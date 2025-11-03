import os, tempfile, shutil, unittest


class TestTrainFramesCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        import sys
        from train_frames import parse_args
        old = sys.argv
        try:
            sys.argv = ["train_frames.py"]
            a = parse_args()
            self.assertEqual(a.data, "data")
            self.assertEqual(a.embed_model, "mobilenetv3_small_100")
            self.assertEqual(a.backbone, "mobilenetv3_small_100")
            self.assertEqual(a.clf, "xgb")
            self.assertEqual(a.alpha, 1.0)
            self.assertEqual(a.hpo_alpha, 10)
            self.assertEqual(a.hpo_xgb, 10)
            # new logreg args
            self.assertEqual(a.C, 1.0)
            self.assertEqual(a.hpo_logreg, 10)
            self.assertEqual(a.logreg_max_iter, 500)
            # Default eval split is 1% to keep fast runs
            self.assertAlmostEqual(a.eval_split, 0.01)
        finally:
            sys.argv = old

    def test_list_videos_discovers_labeled_files(self):
        from vkb.io import list_videos
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "lab1"))
            open(os.path.join(tmp, "lab1", "a.mp4"), "wb").close()
            os.makedirs(os.path.join(tmp, "lab2"))
            open(os.path.join(tmp, "lab2", "b.mp4"), "wb").close()
            vids = list_videos(tmp)
            paths = {p for p,_ in vids}
            labs = {l for _,l in vids}
            self.assertIn(os.path.join(tmp, "lab1", "a.mp4"), paths)
            self.assertIn(os.path.join(tmp, "lab2", "b.mp4"), paths)
            self.assertEqual(labs, {"lab1","lab2"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_clf_display_name_mapping(self):
        from train_frames import _clf_display_name
        self.assertEqual(_clf_display_name("xgb"), "xgboost")
        self.assertEqual(_clf_display_name("ridge"), "ridge")

    def test_hpo_ridge_returns_positive_alpha_and_score(self):
        import numpy as np
        from train_frames import _hpo_ridge
        # linearly separable synthetic data
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)
        a, s, trials = _hpo_ridge(X, y, iters=5, seed=1)
        self.assertTrue(a > 0)
        self.assertTrue(0 <= s <= 1)
        self.assertEqual(len(trials), 5)


if __name__ == "__main__":
    unittest.main()
