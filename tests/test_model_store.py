import os, tempfile, shutil, unittest


class TestModelStore(unittest.TestCase):
    def test_save_and_tags_and_load(self):
        from vkb.model_store import ModelStore
        tmp = tempfile.mkdtemp(); models = os.path.join(tmp, 'models')
        try:
            ms = ModelStore(base_dir=models)
            obj1 = {"id": 1}
            p1 = ms.save(obj1, ["clf", "modelA"], meta={"m":1}, tags=["latest","prod"])  # also writes LATEST via artifacts
            # Pointers exist
            with open(os.path.join(models, 'LATEST')) as f: latest_bn = f.read().strip()
            with open(os.path.join(models, 'PROD')) as f: prod_bn = f.read().strip()
            assert os.path.basename(p1) == latest_bn == prod_bn
            # Load by tags
            o_latest = ms.load('latest'); o_prod = ms.load('prod')
            assert o_latest == obj1 and o_prod == obj1

            # Save another and retag prod
            obj2 = {"id": 2}
            p2 = ms.save(obj2, ["clf", "modelB"], meta={"m":2}, tags=["prod"])  # prod now points to new
            with open(os.path.join(models, 'PROD')) as f: prod2_bn = f.read().strip()
            assert os.path.basename(p2) == prod2_bn
            assert ms.load('prod') == obj2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

