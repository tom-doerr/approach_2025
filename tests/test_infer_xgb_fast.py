import unittest
import numpy as np


class DummyBooster:
    def __init__(self, out):
        self._out = np.array(out, dtype=np.float32)
    def inplace_predict(self, x):
        return self._out


class DummyXGB:
    def __init__(self, out):
        self._booster = DummyBooster(out)
    def get_booster(self):
        return self._booster


class TestXGBFastPredict(unittest.TestCase):
    def test_binary(self):
        from infer_live import _xgb_fast_predict
        clf = DummyXGB([0.8])
        self.assertEqual(_xgb_fast_predict(clf, [0.0, 1.0]), 1)
        clf = DummyXGB([0.2])
        self.assertEqual(_xgb_fast_predict(clf, [0.0, 1.0]), 0)

    def test_multi(self):
        from infer_live import _xgb_fast_predict
        clf = DummyXGB([0.1, 0.7, 0.2])
        self.assertEqual(_xgb_fast_predict(clf, [1.0, 2.0]), 1)


if __name__ == '__main__':
    unittest.main()

