import io, sys, unittest


class TestSampleIndexEdges(unittest.TestCase):
    def setUp(self):
        # Patch list_videos to produce two labels with different frame counts
        import vkb.io as vio
        self.orig_lv = vio.list_videos
        self.vids = [("vid_A", "A"), ("vid_B", "B"), ("vid_B2", "B")]
        vio.list_videos = lambda _root: self.vids

        # Patch cv2 to simulate fixed frame counts per video
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
        self.old_cv2 = sys.modules.get('cv2')
        self.FakeCv2 = types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=FakeCap,
            resize=lambda img, sz: img,
            cvtColor=lambda img, code: img,
            COLOR_BGR2RGB=0,
        )
        sys.modules['cv2'] = self.FakeCv2

    def tearDown(self):
        import vkb.io as vio
        vio.list_videos = self.orig_lv
        if self.old_cv2 is not None: sys.modules['cv2'] = self.old_cv2
        else: del sys.modules['cv2']

    def test_eval_frac_zero_all_train(self):
        import vkb.finetune as vf
        idx = vf.SampleIndex("ignored").load()
        tr, va = idx.tail_split(0.0)
        assert len(va) == 0
        assert len(tr) == len(idx.samples)

    def test_small_frac_min_one_val_per_class(self):
        import vkb.finetune as vf
        idx = vf.SampleIndex("ignored").load()
        tr, va = idx.tail_split(0.01)
        # Build per-class counts from idx.samples
        from collections import defaultdict
        per_class_total = defaultdict(int)
        per_class_val = defaultdict(int)
        for i, (_vp, _fi, ci) in enumerate(idx.samples):
            per_class_total[ci] += 1
        for i in va:
            ci = idx.samples[i][2]
            per_class_val[ci] += 1
        for ci, total in per_class_total.items():
            if total > 1:
                assert per_class_val[ci] >= 1
                assert per_class_val[ci] <= total - 1
        # sanity: disjoint
        assert set(tr).isdisjoint(set(va))

