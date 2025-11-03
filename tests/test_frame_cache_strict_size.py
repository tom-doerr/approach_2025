import os, sys, json, tempfile, shutil
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


def test_truncated_memmap_triggers_rebuild(tmp_path, monkeypatch):
    import vkb.cache as vc
    # Redirect frames cache root
    monkeypatch.setattr(vc, 'frames_root', lambda: str(tmp_path / 'frames'))
    # Fake cv2 (auto-restored)
    monkeypatch.setitem(sys.modules, 'cv2', _fake_cv(h=4, w=5, n=2))

    # Create a fake video file path and source stats
    vid = tmp_path / 'vidA.mp4'
    vid.write_bytes(b'0'*123)  # just to have size/mtime

    # First build (should build fresh)
    meta, built = vc.ensure_frames_cached(str(tmp_path), str(vid))
    assert built is True

    meta_path, data_path = vc.frames_paths(str(tmp_path), str(vid))
    with open(meta_path, 'r') as f:
        meta2 = json.load(f)
    exp_bytes = int(meta2['n']) * int(meta2['h']) * int(meta2['w']) * 3
    assert os.path.getsize(data_path) == exp_bytes

    # Truncate data by 1 byte to simulate corruption
    os.truncate(data_path, exp_bytes - 1)

    # Second call must detect exact-size mismatch and rebuild
    meta3, built2 = vc.ensure_frames_cached(str(tmp_path), str(vid))
    assert built2 is True
    assert os.path.getsize(data_path) == exp_bytes
