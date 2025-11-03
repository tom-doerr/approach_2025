import os, sys, tempfile


def test_record_asks_for_highest_resolution(monkeypatch):
    calls = {"vw_size": None, "set": []}

    class FakeCap:
        def __init__(self, idx):
            self.max_w, self.max_h = 1920, 1080
            self._frames = 3
        def isOpened(self):
            return True
        def set(self, prop, val):
            calls["set"].append((prop, val))
            # ignore exact value; backend will clamp to max
            return True
        def read(self):
            import numpy as np
            if self._frames:
                self._frames -= 1
                return True, np.zeros((self.max_h, self.max_w, 3), dtype=np.uint8)
            return False, None
        def release(self):
            pass

    class FakeWriter:
        def __init__(self, _path, _fourcc, _fps, size):
            calls["vw_size"] = tuple(size)
        def write(self, fr):
            pass
        def release(self):
            pass

    class FakeCv:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        def VideoCapture(self, idx): return FakeCap(idx)
        VideoWriter = FakeWriter
        @staticmethod
        def VideoWriter_fourcc(*a):
            return 0
        @staticmethod
        def imshow(*a, **k):
            pass
        @staticmethod
        def waitKey(*a, **k):
            return 27  # stop immediately
        @staticmethod
        def destroyAllWindows():
            pass

    old_cv = sys.modules.get('cv2')
    sys.modules['cv2'] = FakeCv()
    try:
        from record_video import record
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            record("hires", preview=True)
    finally:
        os.chdir(cwd)
        if old_cv is not None:
            sys.modules['cv2'] = old_cv
        else:
            sys.modules.pop('cv2', None)

    # Writer should be initialized at the camera's max size
    assert calls["vw_size"] == (1920, 1080)
    # We attempted to set width/height at least once
    props = [p for p, _ in calls["set"]]
    assert FakeCv.CAP_PROP_FRAME_WIDTH in props and FakeCv.CAP_PROP_FRAME_HEIGHT in props

