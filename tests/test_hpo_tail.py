import unittest
import numpy as np


class TestHPOTail(unittest.TestCase):
    def test_hpo_tail_runs_with_idx_by_class(self):
        from train_frames import _hpo_ridge
        # Build simple separable data: two classes in 1D
        X = np.concatenate([np.zeros((10,1)), np.ones((10,1))], axis=0)
        y = np.array([0]*10 + [1]*10)
        idx = {0: list(range(0,10)), 1: list(range(10,20))}
        a, s, trials = _hpo_ridge(X, y, iters=3, seed=0, idx_by_class=idx, eval_frac=0.3)
        self.assertTrue(a > 0)
        self.assertEqual(len(trials), 3)


if __name__ == '__main__':
    unittest.main()

