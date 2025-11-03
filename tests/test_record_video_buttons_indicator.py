import os, sys, tempfile, unittest


class TestRecordVideoButtonsIndicator(unittest.TestCase):
    def test_overlay_shows_active_label(self):
        calls = {"texts": []}

        class FakeCap:
            def __init__(self, *_): self._frames = 6
            def isOpened(self): return True
            def read(self):
                import numpy as np
                if self._frames:
                    self._frames -= 1
                    return True, np.zeros((80,160,3), dtype=np.uint8)
                return False, None
            def get(self, *_): return 30.0
            def release(self): pass

        class FakeWriter:
            def __init__(self, *a, **k): pass
            def write(self, _): pass
            def release(self): pass

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
            def putText(self, img, txt, *a, **k): calls["texts"].append(txt)
            def imshow(self, *a, **k): pass
            def waitKey(self, *a, **k): return ord('q')
            def destroyAllWindows(self): pass

        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        from record_video import record_buttons, _layout_buttons
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cwd = os.getcwd(); os.chdir(tmp)
                cv = sys.modules['cv2']
                # Click A once to start recording, then exit
                orig_wait = cv.waitKey
                did_click = {"v": False}
                steps = {"n": 0}
                def waitKey_click_then_quit(*a, **k):
                    if steps["n"] == 0:
                        boxes = _layout_buttons(["A","STOP"], 160, 80)
                        lab, x0, y0, x1, y1 = boxes[0]
                        cv._cb(cv.EVENT_LBUTTONDOWN, x0+2, y0+2, 0, None)
                        steps["n"] += 1
                        return 0
                    elif steps["n"] < 4:
                        steps["n"] += 1
                        return 0
                    return orig_wait(*a, **k)
                cv.waitKey = waitKey_click_then_quit
                record_buttons("A", cam_index=0)
                # We should have drawn an overlay containing the active label, mm:ss, and frame count ("f")
                assert any(("REC:" in t and "A" in t and ":" in t and "f" in t) for t in calls["texts"]) 
                os.chdir(cwd)
        finally:
            if old_cv is not None:
                sys.modules['cv2'] = old_cv
            else:
                del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()
