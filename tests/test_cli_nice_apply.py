def test_maybe_nice_calls_os(monkeypatch):
    import types, importlib, sys
    tf = importlib.import_module('train_frames')
    called = {"val": None}

    def fake_nice(n):
        called["val"] = int(n)

    monkeypatch.setattr(sys.modules['os'], 'nice', fake_nice)

    class A: pass
    a = A(); a.nice = 7
    tf._maybe_nice(a)
    assert called["val"] == 7

