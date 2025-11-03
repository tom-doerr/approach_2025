import os, sys, tempfile, unittest


class TestRecordVideoButtonsToggle(unittest.TestCase):
    def test_click_same_label_toggles_stop(self):
        events = {"released": 0, "path": None}

        class FakeCap:
            def __init__(self, *_): self._frames = 3
            def isOpened(self): return True
            def read(self):
                import numpy as np
                if self._frames:
                    self._frames -= 1
                    return True, np.zeros((100,200,3), dtype=np.uint8)
                return False, None
            def get(self, *_): return 30.0
            def release(self): pass

        class FakeWriter:
            def __init__(self, path, *_, **__): events["path"] = path
            def write(self, _): pass
            def release(self): events["released"] += 1

        class FakeCv:
            VideoCapture = FakeCap
            VideoWriter = FakeWriter
            EVENT_LBUTTONDOWN = 1
            FONT_HERSHEY_SIMPLEX = 0
            def __init__(self): self._cb = None
            def VideoWriter_fourcc(self, *a): return 0
            def namedWindow(self, w): pass
            def setMouseCallback(self, w, cb): self._cb = cb
            def rectangle(self, *a, **k): pass
            def putText(self, *a, **k): pass
            def imshow(self, *a, **k): pass
            def waitKey(self, *a, **k): return ord('q')
            def destroyAllWindows(self): pass

        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        from record_video import record_buttons, _layout_buttons
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cwd = os.getcwd(); os.chdir(tmp)
                cv = sys.modules['cv2']
                orig_wait = cv.waitKey
                clicks = 0
                def waitKey_click_label_twice(*a, **k):
                    nonlocal clicks
                    boxes = _layout_buttons(["A","B","STOP"], 200, 100)
                    if clicks == 0:
                        # click label A to start
                        lab, x0, y0, x1, y1 = boxes[0]
                        cv._cb(cv.EVENT_LBUTTONDOWN, x0+2, y0+2, 0, None)
                    elif clicks == 1:
                        # click A again to stop
                        lab, x0, y0, x1, y1 = boxes[0]
                        cv._cb(cv.EVENT_LBUTTONDOWN, x0+2, y0+2, 0, None)
                    clicks += 1
                    if clicks < 3:
                        return 0
                    return orig_wait(*a, **k)
                cv.waitKey = waitKey_click_label_twice
                record_buttons("A,B", cam_index=0)
                self.assertIsNotNone(events["path"])  # writer created
                self.assertGreaterEqual(events["released"], 1)  # toggled stop released writer
                os.chdir(cwd)
        finally:
            if old_cv is not None:
                sys.modules['cv2'] = old_cv
            else:
                del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()

