import os, sys, tempfile, unittest


class TestRecordVideoHeadless(unittest.TestCase):
    def test_no_preview_does_not_call_destroy(self):
        # Fake cv2 with no GUI; destroyAllWindows would raise if called
        calls = {"destroy": 0}

        class FakeCap:
            def __init__(self, idx):
                self._frames = 2
            def isOpened(self):
                return True
            def read(self):
                if self._frames:
                    self._frames -= 1
                    import numpy as np
                    return True, (np.zeros((2,2,3), dtype=np.uint8))
                return False, None
            def release(self):
                pass

        class FakeWriter:
            def __init__(self, *a, **k):
                pass
            def write(self, fr):
                pass
            def release(self):
                pass

        class FakeCv:
            VideoCapture = FakeCap
            VideoWriter = FakeWriter
            @staticmethod
            def VideoWriter_fourcc(*a):
                return 0
            @staticmethod
            def destroyAllWindows():
                calls["destroy"] += 1
                raise RuntimeError("should not be called")

        old_cv = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv()
        from record_video import record
        try:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                record("headless", preview=False)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass
            if old_cv is not None:
                sys.modules['cv2'] = old_cv
            else:
                del sys.modules['cv2']
        self.assertEqual(calls["destroy"], 0)


if __name__ == "__main__":
    unittest.main()
