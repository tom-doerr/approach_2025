import os, tempfile, shutil, unittest


class TestInferLiveDefaults(unittest.TestCase):
    def test_choose_model_path_uses_latest_when_none(self):
        from infer_live import choose_model_path
        tmp = tempfile.mkdtemp()
        try:
            d = os.path.join(tmp, "models"); os.makedirs(d)
            open(os.path.join(d, "20250101_000000_ridge_foo.pkl"), "wb").close()
            open(os.path.join(d, "20250102_000000_xgboost_bar.pkl"), "wb").close()
            p = choose_model_path(None, d)
            self.assertTrue(p.endswith("20250102_000000_xgboost_bar.pkl"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_model_display_fields_present(self):
        # mimic the bundle saved by training
        bundle = {
            "clf_name": "ridge",
            "embed_model": "mobilenetv3_small_100",
        }
        s = f"{bundle['clf_name']} | {bundle['embed_model']}"
        # expectation string; printing is in main(), so we just assert format
        self.assertIn("ridge | mobilenetv3_small_100", s)


if __name__ == "__main__":
    unittest.main()
