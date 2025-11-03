import os, sys, tempfile, unittest


class TestRecordVideoButtons(unittest.TestCase):
    def test_click_button_starts_recording_in_label_dir(self):
        # Arrange fake cv2 with mouse callback capture
        created = {"path": None}

        class FakeCap:
            def __init__(self, *_):
                self._frames = 2
            def isOpened(self): return True
            def read(self):
                if self._frames:
                    self._frames -= 1
                    import numpy as np
                    return True, (np.zeros((100,200,3), dtype=np.uint8))
                return False, None
            def get(self, *_): return 30.0
            def release(self): pass

        class FakeWriter:
            def __init__(self, path, *_, **__): created["path"] = path
            def write(self, _): pass
            def release(self): pass

        class FakeCv:
            VideoCapture = FakeCap
            VideoWriter = FakeWriter
            EVENT_LBUTTONDOWN = 1
            FONT_HERSHEY_SIMPLEX = 0
            def __init__(self): self._cb = None; self._win=None
            def VideoWriter_fourcc(self, *a): return 0
            def namedWindow(self, w): self._win = w
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
                # Trigger: call record_buttons; before waitKey returns, simulate click
                # Install a small wrapper to call the stored callback once
                cv = sys.modules['cv2']
                path_before = created["path"]
                # Call record_buttons in a way that triggers one frame loop
                # We need to invoke the click after the window and callback are set up
                # Hack: monkeypatch waitKey to fire click first, then return 'q'
                original_waitKey = cv.waitKey
                def waitKey_once(*a, **k):
                    # Compute a point inside first button
                    boxes = _layout_buttons(["A","B"], 200, 100)
                    lab, x0, y0, x1, y1 = boxes[0]
                    x = x0 + 5; y = y0 + 5
                    if hasattr(cv, '_cb') and cv._cb:
                        cv._cb(cv.EVENT_LBUTTONDOWN, x, y, 0, None)
                    cv.waitKey = original_waitKey
                    return original_waitKey(*a, **k)
                cv.waitKey = waitKey_once
                record_buttons("A,B", cam_index=0)
                self.assertIsNotNone(created["path"])  # writer created
                self.assertIn(os.path.join("data","A"), created["path"])  # label folder
                os.chdir(cwd)
        finally:
            if old_cv is not None:
                sys.modules['cv2'] = old_cv
            else:
                del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()

