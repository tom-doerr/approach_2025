import io, os, sys, tempfile


def test_classic_prints_newest_video(monkeypatch):
    from train_frames import _prepare_classic_io
    # Make a temp data root with two videos of different mtime
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, 'data')
        os.makedirs(os.path.join(data, 'A'), exist_ok=True)
        v1 = os.path.join(data, 'A', '20250101_000000.mp4')
        v2 = os.path.join(data, 'A', '20251212_235959.mp4')
        open(v1, 'wb').close(); open(v2, 'wb').close()
        os.utime(v1, (1, 1)); os.utime(v2, (2, 2))
        # capture stdout
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:
            from types import SimpleNamespace
            args = SimpleNamespace(data=data, allow_test_labels=True)
            _prepare_classic_io(args)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        assert 'Newest video:' in out and '20251212_235959.mp4' in out


def test_finetune_prepare_prints_newest(monkeypatch):
    import vkb.finetune as ft
    # Stub _build_samples to avoid cv2
    monkeypatch.setattr(ft, '_build_samples', lambda vids, lab2i, stride=1, stride_offset=0: [(vids[0][0], 0, 0)] if vids else [])
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, 'data')
        os.makedirs(os.path.join(data, 'A'), exist_ok=True)
        os.makedirs(os.path.join(data, 'B'), exist_ok=True)
        a1 = os.path.join(data, 'A', '20240101_000000.mp4')
        v1 = os.path.join(data, 'B', '20250101_000000.mp4')
        v2 = os.path.join(data, 'B', '20251212_235959.mp4')
        for p in (a1, v1, v2): open(p, 'wb').close()
        os.utime(a1, (1,1)); os.utime(v1, (2,2)); os.utime(v2, (3,3))
        # Make _labels_and_videos list our three files (two classes)
        monkeypatch.setattr(ft, '_labels_and_videos', lambda root: ([(a1,'A'), (v1, 'B'), (v2, 'B')], ['A','B'], {'A':0,'B':1}))
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        try:
            from types import SimpleNamespace
            cons = type('C', (), {'print': lambda self, *a, **k: print(*a, **k)})()
            args = SimpleNamespace(data=data, eval_split=0.0, test_split=0.0)
            ft._prepare_data(args, cons)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        assert 'Newest video:' in out and '20251212_235959.mp4' in out
