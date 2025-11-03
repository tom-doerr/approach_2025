import unittest
import numpy as np


class TestHPOLogRegTail(unittest.TestCase):
    def test_hpo_logreg_tail(self):
        from train_frames import _hpo_logreg
        X = np.concatenate([np.zeros((20,3)), np.ones((20,3))], axis=0)
        y = np.array([0]*20 + [1]*20)
        idx = {0: list(range(0,20)), 1: list(range(20,40))}
        C, s, trials = _hpo_logreg(X, y, iters=3, seed=0, idx_by_class=idx, eval_frac=0.3)
        self.assertTrue(C > 0)
        self.assertEqual(len(trials), 3)


if __name__ == '__main__':
    unittest.main()

