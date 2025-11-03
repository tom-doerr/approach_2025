import numpy as np
import unittest


class TestPorts(unittest.TestCase):
    def test_embedder_protocol(self):
        from vkb.ports import Embedder

        class E:
            def __call__(self, frame: np.ndarray) -> np.ndarray:
                return frame.mean(axis=(0, 1)) if frame.ndim == 3 else np.asarray(frame)

        e = E()
        assert isinstance(e, Embedder)
        out = e(np.zeros((2, 2, 3), dtype=np.uint8))
        assert isinstance(out, np.ndarray)

    def test_cache_adapter_delegates(self):
        calls = []

        class FakeCacheModule:
            def ensure_frames_cached(self, data_root, video_path):
                calls.append(("ensure", data_root, video_path))

            def open_frames_memmap(self, data_root, video_path):
                calls.append(("open", data_root, video_path))
                return np.zeros((1, 2, 2, 3), dtype=np.uint8), {"n": 1, "h": 2, "w": 2}

        from vkb.adapters import CacheModuleAdapter
        from vkb.ports import FrameCache

        adp = CacheModuleAdapter(FakeCacheModule())
        assert isinstance(adp, FrameCache)
        adp.ensure_frames_cached("root", "vid.mp4")
        arr, meta = adp.open_frames_memmap("root", "vid.mp4")
        assert arr.shape[0] == 1 and meta["n"] == 1
        assert calls[0][0] == "ensure" and calls[1][0] == "open"

    def test_augmenter_protocol(self):
        from vkb.ports import Augmenter

        class A:
            def apply(self, img: np.ndarray) -> np.ndarray:
                return img

        a = A()
        assert isinstance(a, Augmenter)

    def test_console_protocol(self):
        from vkb.ports import Console

        class C:
            def __init__(self): self.buf = []
            def print(self, *args, **kw): self.buf.append(args)

        c = C()
        assert isinstance(c, Console)
        c.print("hello")
        assert c.buf


if __name__ == "__main__":
    unittest.main()

