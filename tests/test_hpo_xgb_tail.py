import unittest
import numpy as np


class TestHPOXGBTail(unittest.TestCase):
    def test_hpo_xgb_tail_runs(self):
        from train_frames import _hpo_xgb
        X = np.concatenate([np.zeros((20,2)), np.ones((20,2))], axis=0)
        y = np.array([0]*20 + [1]*20)
        idx = {0: list(range(0,20)), 1: list(range(20,40))}
        p, s, trials = _hpo_xgb(X, y, iters=2, seed=0, idx_by_class=idx, eval_frac=0.3)
        self.assertIsInstance(p, dict)
        self.assertTrue(len(trials) == 2)


if __name__ == '__main__':
    unittest.main()

