import unittest


class TestInferFormat(unittest.TestCase):
    def test_format_timings(self):
        from infer_live import _format_timings
        s = _format_timings(37.49, 8.26, 5.73, 1.11)
        self.assertIn('FPS: 37.5', s)
        self.assertIn('Proc: 8.3 ms', s)
        self.assertIn('Emb: 5.7 ms', s)
        self.assertIn('Clf: 1.1 ms', s)


if __name__ == '__main__':
    unittest.main()

