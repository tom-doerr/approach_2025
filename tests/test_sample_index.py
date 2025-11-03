import io, sys, os, shutil, tempfile, unittest


class TestSampleIndex(unittest.TestCase):
    def test_index_load_and_tail_split(self):
        import vkb.finetune as vf

        # Patch list_videos to return two labels with different videos
        vids = [("vid_A", "A"), ("vid_B", "B"), ("vid_B2", "B")]
        import vkb.io as vio
        orig_lv = vio.list_videos
        vio.list_videos = lambda _root: vids

        # Fake cv2 with deterministic frame counts
        import numpy as np, types
        class FakeCap:
            def __init__(self, path):
                n = 8 if path.endswith("A") else 12
                self.frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
            def get(self, prop): return len(self.frames)
            def set(self, prop, value): pass
        FakeCv2 = types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=FakeCap,
            resize=lambda img, sz: img,
            cvtColor=lambda img, code: img,
            COLOR_BGR2RGB=0,
        )
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2

        try:
            idx = vf.SampleIndex("ignored").load()
            assert idx.labels == ["A", "B"]
            # counts: A has 8 frames; B has 12+12 = 24
            assert idx.counts() == [8, 24]
            tr, va = idx.tail_split(0.25)
            # ensure non-empty split and disjoint
            assert len(tr) > 0 and len(va) > 0
            assert set(tr).isdisjoint(set(va))
            # parity with direct helper
            tr2, va2 = vf._tail_split(idx.samples, len(idx.labels), 0.25)
            assert tr == tr2 and va == va2
        finally:
            vio.list_videos = orig_lv
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']


if __name__ == "__main__":
    unittest.main()

