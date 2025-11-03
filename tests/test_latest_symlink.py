import os, sys, tempfile, unittest


class TestLatestSymlink(unittest.TestCase):
    def test_symlink_created_and_updated(self):
        import numpy as np
        from record_video import record

        class FakeCap:
            def __init__(self, idx):
                self._frames = 2
            def isOpened(self):
                return True
            def read(self):
                if self._frames:
                    self._frames -= 1
                    return True, np.zeros((4,4,3), dtype=np.uint8)
                return False, None
            def release(self):
                pass

        class FakeWriter:
            def __init__(self, *a, **k):
                self.path = a[0] if a else ""
                open(self.path, "ab").close()  # ensure file exists
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
                pass

        old = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv()
        try:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                record("lab", preview=False)
                labdir = os.path.join("data", "lab")
                link = os.path.join(labdir, "latest")
                self.assertTrue(os.path.islink(link))
                first_target = os.readlink(link)
                import time; time.sleep(1.1)
                record("lab", preview=False)
                second_target = os.readlink(link)
                self.assertNotEqual(first_target, second_target)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass
            if old is not None:
                sys.modules['cv2'] = old
            else:
                del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()
