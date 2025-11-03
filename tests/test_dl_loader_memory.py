import os, sys, tempfile, shutil, unittest


class TestDLLoaderMemory(unittest.TestCase):
    def test_iter_no_workers(self):
        # Build tiny dataset and iterate with workers=0
        import numpy as np
        from vkb.cache import ensure_frames_cached, open_frames_memmap
        from vkb.finetune import FrameDataset
        from torch.utils.data import DataLoader

        tmp = tempfile.mkdtemp(); data_root = os.path.join(tmp, 'data'); os.makedirs(os.path.join(data_root, 'lab'))
        vp = os.path.join(data_root, 'lab', 'a.mp4'); open(vp, 'wb').close()

        class FakeCap:
            def __init__(self, path): self.frames = [np.zeros((8,8,3), dtype=np.uint8) for _ in range(4)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)
            def release(self): pass
            def set(self, *a): pass
        class FakeCv:
            VideoCapture = FakeCap
            def resize(self, img, sz): return img
            def cvtColor(self, img, code): return img
            COLOR_BGR2RGB = 0
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        try:
            ensure_frames_cached(data_root, vp)
            ds = FrameDataset([(vp, i, 0) for i in range(4)], img_size=8, data_root=data_root)
            dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            it = iter(dl)
            xb, yb = next(it)
            assert xb.shape[0] == 2
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)

    def test_iter_workers_file_system(self):
        # Small run with 1 worker and file_system sharing
        import numpy as np, torch
        from vkb.cache import ensure_frames_cached
        from vkb.finetune import FrameDataset
        from torch.utils.data import DataLoader

        tmp = tempfile.mkdtemp(); data_root = os.path.join(tmp, 'data'); os.makedirs(os.path.join(data_root, 'lab'))
        vp = os.path.join(data_root, 'lab', 'b.mp4'); open(vp, 'wb').close()
        class FakeCap:
            def __init__(self, path): self.frames = [np.zeros((6,6,3), dtype=np.uint8) for _ in range(4)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop(0)) if self.frames else (False, None)
            def release(self): pass
            def set(self, *a): pass
        class FakeCv:
            VideoCapture = FakeCap
            def resize(self, img, sz): return img
            def cvtColor(self, img, code): return img
            COLOR_BGR2RGB = 0
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
        try:
            torch.multiprocessing.set_sharing_strategy('file_system')
            ensure_frames_cached(data_root, vp)
            ds = FrameDataset([(vp, i, 0) for i in range(4)], img_size=6, data_root=data_root)
            dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=1, prefetch_factor=1, persistent_workers=False)
            it = iter(dl); xb, yb = next(it)
            assert xb.shape[0] == 2
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
