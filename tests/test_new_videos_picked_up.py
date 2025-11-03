import sys


def test_prepare_data_counts_increase_with_new_videos(monkeypatch, tmp_path):
    # Arrange: two lists of videos, then add a third newer one
    from vkb import finetune as ft
    vids0 = [(str(tmp_path/"A1.mp4"), "A"), (str(tmp_path/"B1.mp4"), "B")]
    vids1 = vids0 + [(str(tmp_path/"A2.mp4"), "A")]

    # Create dummy files so list_videos glob would find them (we monkeypatch anyway)
    for p, _ in vids1:
        open(p, "wb").close()

    # Fake cv2 that reports frame counts (A1=5, B1=6, A2=4)
    class FakeCap:
        def __init__(self, path):
            self.path = path
            self._open = True
            if path.endswith("A1.mp4"): self._n = 5
            elif path.endswith("B1.mp4"): self._n = 6
            else: self._n = 4
            # also provide frames for fallback counting
            import numpy as np
            self._frames = [np.zeros((2,2,3), dtype=np.uint8)] * self._n
        def isOpened(self): return True
        def get(self, prop): return self._n
        def read(self):
            if self._frames:
                import numpy as np
                return True, self._frames.pop()
            return False, None
        def release(self): pass

    class FakeCv: 
        CAP_PROP_FRAME_COUNT = 7
        def VideoCapture(self, path): return FakeCap(path)

    old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
    try:
        monkeypatch.setattr(ft, '_labels_and_videos', lambda root: (vids0, ["A","B"], {"A":0, "B":1}))
        vids, labels, lab2i = ft._labels_and_videos(str(tmp_path))
        s0 = ft._build_samples(vids, lab2i)
        # Now include the new video and recompute
        monkeypatch.setattr(ft, '_labels_and_videos', lambda root: (vids1, ["A","B"], {"A":0, "B":1}))
        vids, labels, lab2i = ft._labels_and_videos(str(tmp_path))
        s1 = ft._build_samples(vids, lab2i)
        assert len(s1) - len(s0) == 4  # A2 contributes 4 frames
    finally:
        if old_cv is not None: sys.modules['cv2'] = old_cv
        else: sys.modules.pop('cv2', None)

