import sys


def test_cli_nice_default_is_10():
    from train_frames import parse_args
    old = sys.argv
    try:
        sys.argv = ["train_frames.py"]
        a = parse_args()
        assert int(a.nice) == 10
    finally:
        sys.argv = old

