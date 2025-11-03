import sys, tempfile
import numpy as np


def _fake_cv(h=8, w=8, n=3):
    class FakeCap:
        def __init__(self, *_):
            self._frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
        def isOpened(self): return True
        def read(self):
            if self._frames:
                return True, self._frames.pop(0)
            return False, None
        def release(self): pass
    class CV:
        VideoCapture = FakeCap
    return CV()


def test_dims_mismatch_triggers_rebuild(tmp_path, monkeypatch):
    import vkb.cache as vc
    # redirect cache root
    monkeypatch.setattr(vc, 'frames_root', lambda: str(tmp_path / 'frames'))
    vid = tmp_path / 'vidB.mp4'
    vid.write_bytes(b'1')

    # initial build with 4x5
    monkeypatch.setitem(sys.modules, 'cv2', _fake_cv(h=4, w=5, n=2))
    meta1, built1 = vc.ensure_frames_cached(str(tmp_path), str(vid))
    assert built1 is True

    # change reported dims to 6x5; expect rebuild
    monkeypatch.setitem(sys.modules, 'cv2', _fake_cv(h=6, w=5, n=2))
    meta2, built2 = vc.ensure_frames_cached(str(tmp_path), str(vid))
    assert built2 is True

