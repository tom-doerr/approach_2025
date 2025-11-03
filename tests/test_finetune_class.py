import unittest


class TestFinetunerClass(unittest.TestCase):
    def test_class_exists_and_run_delegates(self):
        import types
        import vkb.finetune as vf
        assert hasattr(vf, 'Finetuner')

        called = {}
        orig = vf.finetune
        try:
            vf.finetune = lambda args: called.setdefault('ok', True) or None
            ft = vf.Finetuner(types.SimpleNamespace())
            ft.run()
            assert called.get('ok') is True
        finally:
            vf.finetune = orig


if __name__ == "__main__":
    unittest.main()

