import io, os, sys, tempfile


def test_record_prints_using_mode(monkeypatch):
    class FakeCap:
        def __init__(self, idx):
            self._w, self._h, self._fps = 1280, 720, 60.0
            self._reads = 0
        def isOpened(self): return True
        def getBackendName(self): return "FAKE"
        def set(self, prop, val): return True
        def get(self, prop):
            if prop == 3: return float(self._w)
            if prop == 4: return float(self._h)
            if prop == 5: return float(self._fps)
            return 0.0
        def read(self):
            import numpy as np
            self._reads += 1
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
        from record_video import record
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                record("mode", preview=True)
        finally:
            sys.stdout = old
            try: os.chdir(cwd)
            except Exception: pass
        out = buf.getvalue()
        assert 'Using:' in out and ('1280x720@60' in out or '1280x720@59' in out)
    finally:
        if old_cv is not None: sys.modules['cv2'] = old_cv
        else: sys.modules.pop('cv2', None)

