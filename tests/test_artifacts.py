import os, tempfile, shutil, unittest


class TestArtifacts(unittest.TestCase):
    def test_save_model_writes_timestamped_file(self):
        from vkb.artifacts import save_model
        tmp = tempfile.mkdtemp()
        try:
            p = save_model({"a":1}, ["ridge", "mobilenetv3_small_100"], base_dir=os.path.join(tmp, "artifacts"))
            self.assertTrue(p.endswith(".pkl"))
            self.assertIn("ridge_mobilenetv3_small_100", os.path.basename(p))
            self.assertTrue(os.path.exists(p))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_latest_model_picks_last_by_name(self):
        from vkb.artifacts import latest_model
        tmp = tempfile.mkdtemp()
        try:
            d = os.path.join(tmp, "models"); os.makedirs(d)
            open(os.path.join(d, "20240101_000000_ridge_mnv3.pkl"), "wb").close()
            open(os.path.join(d, "20250101_000000_xgboost_mnv3.pkl"), "wb").close()
            p = latest_model(d)
            self.assertTrue(p.endswith("20250101_000000_xgboost_mnv3.pkl"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
