import unittest


class TestCLIMpStrideDefault(unittest.TestCase):
    def test_default_mp_stride_is_one(self):
        from train_frames import parse_args
        a = parse_args(["--clf","mp_logreg","--data","."])
        assert a.mp_stride == 1


if __name__ == "__main__":
    unittest.main()

