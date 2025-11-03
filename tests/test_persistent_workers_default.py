import unittest


class TestPersistentWorkersDefault(unittest.TestCase):
    def test_default_persistent_workers_is_false(self):
        import sys
        from train_frames import parse_args
        old = sys.argv
        try:
            sys.argv = ["train_frames.py", "--clf", "dl"]
            a = parse_args()
            assert a.persistent_workers is False
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
