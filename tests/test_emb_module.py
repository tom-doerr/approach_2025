import importlib, types, unittest


class TestEmbModule(unittest.TestCase):
    def test_create_embedder_is_callable(self):
        emb = importlib.import_module('vkb.emb')
        self.assertTrue(hasattr(emb, 'create_embedder'))
        self.assertIsInstance(emb.create_embedder, types.FunctionType)


if __name__ == "__main__":
    unittest.main()

