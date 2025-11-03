import sys


def test_main_invokes_thread_caps_and_nice(monkeypatch):
    import importlib
    tf = importlib.import_module('train_frames')

    called = {"caps": 0, "nice": 0, "train": 0}

    def fake_caps(a):
        called["caps"] += 1

    def fake_nice(a):
        called["nice"] += 1

    def fake_train(a):
        called["train"] += 1
        return None

    monkeypatch.setattr(tf, "_apply_thread_caps", fake_caps)
    monkeypatch.setattr(tf, "_maybe_nice", fake_nice)
    monkeypatch.setattr(tf, "train", fake_train)
    # Avoid dataset probing side effects in main()
    monkeypatch.setattr(tf, "list_videos", lambda _root: [])

    old = sys.argv
    try:
        sys.argv = ["train_frames.py"]
        tf.main()
    finally:
        sys.argv = old

    assert called["caps"] == 1
    assert called["nice"] == 1
    assert called["train"] == 1

