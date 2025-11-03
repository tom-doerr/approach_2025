import os, sys, json, tempfile, shutil, unittest


class TestFrameCache(unittest.TestCase):
    def test_build_and_read_cache(self):
        from vkb.cache import ensure_frames_cached, open_frames_memmap, frames_paths
        tmp = tempfile.mkdtemp()
        data_root = os.path.join(tmp, 'data'); os.makedirs(data_root)
        vp = os.path.join(data_root, 'lab', 'vid.mp4'); os.makedirs(os.path.dirname(vp))
        # Fake cv2
        import numpy as np
        class FakeCap:
            def __init__(self, path):
                self.frames = [np.full((4,4,3), 10, dtype=np.uint8), np.full((4,4,3), 20, dtype=np.uint8)]
                self._pos = 0
            def isOpened(self): return True
            def read(self):
                if not self.frames: return False, None
                return True, self.frames.pop(0)
            def get(self, prop):
                # frame count
                return 2
            def release(self): pass
            def set(self, *a): pass
        class FakeCv:
            VideoCapture = FakeCap
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        try:
            meta, built = ensure_frames_cached(data_root, vp)
            self.assertTrue(built)
            self.assertEqual(meta['n'], 2)
            mm, m2 = open_frames_memmap(data_root, vp)
            self.assertEqual(mm.shape, (2,4,4,3))
            self.assertEqual(int(mm[0,0,0,0]), 10)
            self.assertEqual(int(mm[1,0,0,0]), 20)
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

