import unittest


class TestHPOSort(unittest.TestCase):
    def test_sort_ridge_trials(self):
        from train_frames import _sort_ridge_trials
        trials = [(0.1, 0.9), (1.0, 0.8), (0.01, 0.95)]
        sorted_trials = _sort_ridge_trials(trials)
        scores = [s for _, s in sorted_trials]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[-1], max(s for _, s in trials))

    def test_sort_xgb_trials(self):
        from train_frames import _sort_xgb_trials
        trials = [({'md':3}, 0.7), ({'md':4}, 0.85), ({'md':5}, 0.6)]
        sorted_trials = _sort_xgb_trials(trials)
        scores = [s for _, s in sorted_trials]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[-1], max(s for _, s in trials))


if __name__ == '__main__':
    unittest.main()

