import os, sys, tempfile, shutil, unittest


class TestFrameCacheInvalidate(unittest.TestCase):
    def test_rebuild_on_dim_change(self):
        from vkb.cache import ensure_frames_cached, open_frames_memmap
        tmp = tempfile.mkdtemp(); data_root = os.path.join(tmp, 'data'); os.makedirs(data_root)
        vp = os.path.join(data_root, 'lab', 'vid.mp4'); os.makedirs(os.path.dirname(vp), exist_ok=True)

        import numpy as np
        # First version: 4x4 frames
        class FakeCapA:
            def __init__(self, *_):
                self.frames = [np.full((4,4,3), 11, dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)
            def release(self): pass
            def set(self, *a): pass
        # Second version: 5x5 frames
        class FakeCapB:
            def __init__(self, *_):
                self.frames = [np.full((5,5,3), 22, dtype=np.uint8) for _ in range(3)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)
            def release(self): pass
            def set(self, *a): pass

        class FakeCv:
            VideoCapture = None

        old_cv = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv()
        try:
            sys.modules['cv2'].VideoCapture = FakeCapA
            meta1, built1 = ensure_frames_cached(data_root, vp)
            self.assertTrue(built1)
            self.assertEqual(meta1['h'], 4)
            # change dims; ensure_frames_cached should detect mismatch and rebuild
            sys.modules['cv2'].VideoCapture = FakeCapB
            meta2, built2 = ensure_frames_cached(data_root, vp)
            self.assertTrue(built2)
            self.assertEqual(meta2['h'], 5)
            mm, _ = open_frames_memmap(data_root, vp)
            self.assertEqual(mm.shape[1], 5)
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

