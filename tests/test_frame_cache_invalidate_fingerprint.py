import os, sys, tempfile, shutil, unittest


class TestFrameCacheInvalidateFingerprint(unittest.TestCase):
    def test_rebuild_on_src_file_change(self):
        from vkb.cache import ensure_frames_cached
        tmp = tempfile.mkdtemp(); data_root = os.path.join(tmp, 'data'); os.makedirs(data_root)
        vp = os.path.join(data_root, 'lab', 'vid.mp4'); os.makedirs(os.path.dirname(vp), exist_ok=True)
        # Create a real file so stat() works
        with open(vp, 'wb') as f: f.write(b'abc')

        import numpy as np
        class FakeCap:
            def __init__(self, *_):
                self.frames = [np.full((4,4,3), 1, dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)
            def release(self): pass
            def set(self, *a): pass
        class FakeCv:
            VideoCapture = FakeCap

        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        try:
            meta1, built1 = ensure_frames_cached(data_root, vp)
            self.assertTrue(built1)
            # Touch the source file (change size) → should rebuild
            with open(vp, 'ab') as f: f.write(b'xyz123')
            meta2, built2 = ensure_frames_cached(data_root, vp)
            self.assertTrue(built2)
            self.assertNotEqual(meta1.get('src_size'), meta2.get('src_size'))
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

