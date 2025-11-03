import os, sys, tempfile, unittest


class TestRecordVideoHelpers(unittest.TestCase):
    def test_safe_label(self):
        from vkb.io import safe_label
        self.assertEqual(safe_label("abc-123_X y/z"), "abc-123_X_y_z")

    def test_make_output_path_creates_dir_and_ext(self):
        from vkb.io import make_output_path
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                p = make_output_path("testlabel")
                self.assertTrue(p.startswith(os.path.join("data", "testlabel")))
                self.assertTrue(p.endswith(".mp4"))
                self.assertTrue(os.path.isdir(os.path.join("data", "testlabel")))
            finally:
                os.chdir(cwd)

    def test_parse_args(self):
        from record_video import parse_args
        old = sys.argv
        try:
            sys.argv = ["record_video.py", "--label", "foo"]
            args = parse_args()
            self.assertEqual(args.label, "foo")
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
