import unittest


class TestWorkersDefault(unittest.TestCase):
    def test_default_workers_is_1(self):
        import sys
        from train_frames import parse_args
        old = sys.argv
        try:
            sys.argv = ["train_frames.py", "--clf", "dl"]
            a = parse_args()
            assert a.workers == 1
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
