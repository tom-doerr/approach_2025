import io, os, sys, tempfile


def test_record_lists_capabilities_by_default(monkeypatch):
    import record_video as rv
    # Pretend Linux and inject a fake modes lister
    monkeypatch.setenv('PYTEST_FORCE_LINUX', '1')  # just a marker
    monkeypatch.setattr(sys, 'platform', 'linux', raising=False)
    monkeypatch.setattr(rv, '_list_modes_linux', lambda idx: 'Fake formats list\n')

    class FakeCap:
        def __init__(self, idx): self._w, self._h = 640, 480
        def isOpened(self): return True
        def read(self):
            import numpy as np
            return True, np.zeros((self._h, self._w, 3), dtype=np.uint8)
        def release(self): pass

    class FakeWriter:
        def __init__(self, *a, **k): pass
        def write(self, fr): pass
        def release(self): pass

    class FakeCv:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        def VideoCapture(self, idx): return FakeCap(idx)
        VideoWriter = FakeWriter
        @staticmethod
        def VideoWriter_fourcc(*a): return 0
        @staticmethod
        def imshow(*a, **k): pass
        @staticmethod
        def waitKey(*a, **k): return 27
        @staticmethod
        def destroyAllWindows(): pass

    old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
    try:
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                rv.record('default_list', preview=True, cam_index=0)
        finally:
            sys.stdout = old
            try: os.chdir(cwd)
            except Exception: pass
        out = buf.getvalue()
        assert 'Fake formats list' in out
    finally:
        if old_cv is not None: sys.modules['cv2'] = old_cv
        else: sys.modules.pop('cv2', None)
