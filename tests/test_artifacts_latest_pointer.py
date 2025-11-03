import os, shutil, tempfile, pickle, unittest


class TestArtifactsLatestPointer(unittest.TestCase):
    def test_latest_from_pointer(self):
        from vkb.artifacts import save_model, latest_model
        tmp = tempfile.mkdtemp(); models = os.path.join(tmp, 'models'); os.makedirs(models)
        a = save_model({'a':1}, ['ridge','mobilenetv3_small_100'], base_dir=models)
        b = save_model({'b':2}, ['xgboost','foo'], base_dir=models)
        # latest should be b by pointer
        lm = latest_model(base_dir=models)
        self.assertTrue(lm.endswith(os.path.basename(b)))
        # remove pointer; fallback to sorted
        os.remove(os.path.join(models, 'LATEST'))
        lm2 = latest_model(base_dir=models)
        self.assertTrue(lm2.endswith(os.path.basename(b)))


if __name__ == '__main__':
    unittest.main()
