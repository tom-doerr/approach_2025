import sys, unittest


class TestInferCLI(unittest.TestCase):
    def test_parse_args_defaults_and_path(self):
        from infer_live import parse_args
        old = sys.argv
        try:
            sys.argv = ["infer_live.py"]
            a = parse_args()
            self.assertIsNone(a.model_path)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()

