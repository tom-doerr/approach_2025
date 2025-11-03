import unittest


class TestFinetuneArgs(unittest.TestCase):
    def test_parse_args_has_dl_and_flags(self):
        import sys
        from train_frames import parse_args
        old = sys.argv
        try:
            sys.argv = ["train_frames.py", "--clf", "dl"]
            a = parse_args()
            self.assertEqual(a.clf, "dl")
            self.assertEqual(a.epochs, 10)
            self.assertEqual(a.batch_size, 128)
            self.assertEqual(a.lr, 1e-4)
            self.assertEqual(a.wd, 1e-4)
            self.assertIn(a.device, ("auto","cpu","cuda"))
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
