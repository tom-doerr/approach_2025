import unittest


class TestFmtXGBParams(unittest.TestCase):
    def test_formats_full_names(self):
        from train_frames import _fmt_xgb_params
        p = {
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.025,
            'subsample': 0.83,
            'colsample_bytree': 0.92,
            'reg_lambda': 0.1,
        }
        s = _fmt_xgb_params(p)
        for key in ['max_depth','n_estimators','learning_rate','subsample','colsample_bytree','reg_lambda']:
            self.assertIn(key, s)
        self.assertIn('learning_rate=0.025', s)


if __name__ == '__main__':
    unittest.main()

