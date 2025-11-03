import io, os, unittest


class TestFinetuneHelpersSplit(unittest.TestCase):
    def test_helpers_exist(self):
        # Simple source check to ensure _run_epoch was split
        p = os.path.join(os.path.dirname(__file__), "..", "vkb", "finetune.py")
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        assert "def _train_epoch(" in src
        assert "def _validate_epoch(" in src
        assert "def _print_perf(" in src


if __name__ == "__main__":
    unittest.main()

