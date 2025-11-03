import os, tempfile, shutil, unittest


class TestArtifactsList(unittest.TestCase):
    def test_list_models_sorted(self):
        from vkb.artifacts import list_models
        tmp = tempfile.mkdtemp()
        try:
            d = os.path.join(tmp, "models"); os.makedirs(d)
            a = os.path.join(d, "20240101_000000_a.pkl")
            b = os.path.join(d, "20250101_000000_b.pkl")
            open(a, "wb").close(); open(b, "wb").close()
            paths = list_models(d)
            self.assertEqual(paths, [a, b])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

