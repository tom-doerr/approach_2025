import sys


def test_persistent_workers_default_and_flag_behavior():
    from train_frames import parse_args
    old = sys.argv
    try:
        # default: OFF
        sys.argv = ["train_frames.py"]
        a = parse_args()
        assert a.persistent_workers is False
        assert getattr(a, 'no_persistent_workers', False) is False
        # explicit enable
        sys.argv = ["train_frames.py", "--persistent-workers"]
        b = parse_args()
        assert b.persistent_workers is True
        # explicit disable flag
        sys.argv = ["train_frames.py", "--no-persistent-workers"]
        c = parse_args()
        assert getattr(c, 'no_persistent_workers', False) is True
    finally:
        sys.argv = old

