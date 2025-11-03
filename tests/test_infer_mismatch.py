import unittest


class TestInferMismatch(unittest.TestCase):
    def test_check_feat_dim_raises(self):
        from infer_live import _check_feat_dim
        class C: n_features_in_ = 3
        with self.assertRaises(ValueError) as ctx:
            _check_feat_dim(C(), 5)
        self.assertIn('expects 3, got 5', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()

