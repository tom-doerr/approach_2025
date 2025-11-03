import threading, sys


def test_ensure_frames_cached_concurrent_build(monkeypatch, tmp_path):
    # Stub cv2 to synthesize frames deterministically
    class FakeCv2:
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                # two videos produce different constant frames but same HxW
                val = 10 if path.endswith('A.mp4') else 20
                self.frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(6)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())

    from vkb.cache import ensure_frames_cached, frames_root

    data_root = str(tmp_path)
    # Two paths under tmp; content is irrelevant for stub
    vp = str(tmp_path / 'A.mp4')

    results = []
    barrier = threading.Barrier(2)

    def _worker():
        barrier.wait()
        meta, built = ensure_frames_cached(data_root, vp)
        results.append((meta, built))

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    # At least one build must happen; a second thread may rebuild after lock release
    built_count = sum(1 for _m, b in results if b)
    assert built_count >= 1
    # Both see identical metadata and nonzero frame count
    m0, _ = results[0]; m1, _ = results[1]
    assert int(m0.get('n', 0)) == int(m1.get('n', 0)) > 0
    assert int(m0['h']) == int(m1['h']) == 8 and int(m0['w']) == int(m1['w']) == 8
