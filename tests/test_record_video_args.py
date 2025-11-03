import sys, unittest


class TestRecordVideoArgs(unittest.TestCase):
    def test_no_preview_flag(self):
        from record_video import parse_args
        old = sys.argv
        try:
            sys.argv = ["record_video.py", "--label", "foo", "--no-preview"]
            a = parse_args()
            self.assertTrue(a.no_preview)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()

