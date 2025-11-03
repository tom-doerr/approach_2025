import unittest, sys


class TestVisdomDefaults(unittest.TestCase):
    def test_default_visdom_aug_enabled(self):
        from train_frames import parse_args
        old = sys.argv
        try:
            sys.argv = ["train_frames.py", "--clf", "dl"]
            a = parse_args()
            self.assertGreaterEqual(a.visdom_aug, 1)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()

