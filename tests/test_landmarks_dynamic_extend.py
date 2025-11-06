import os, tempfile, shutil
import numpy as np


def test_landmarks_cache_dynamic_extend(monkeypatch):
    import vkb.landmarks as lm
    tmp = tempfile.mkdtemp()
    try:
        vid = os.path.join(tmp, 'x.mp4'); open(vid, 'wb').close()
        calls = {'n': 0}

        def fake_compute(path, stride, max_frames, start_from=0):
            calls['n'] += 1
            if start_from == 0:
                idx = np.array([0,5,10], dtype=int)
            else:
                idx = np.array([15,20,25], dtype=int)
            pts = np.zeros((len(idx), 21, 3), dtype=float)
            # make non-zero so L1-norm works
            pts[:,0,0] = np.arange(1, len(idx)+1)
            return idx, pts

        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fake_compute)

        # First call builds cache with k=3 (limit)
        f1 = lm.extract_features_for_video(vid, stride=5, max_frames=3)
        assert f1.shape[0] == 3

        # Second call with unlimited should extend (not rebuild from scratch)
        f2 = lm.extract_features_for_video(vid, stride=5, max_frames=0)
        assert f2.shape[0] >= 6
        assert calls['n'] >= 2
    finally:
        # best-effort cleanup of cache file
        try:
            from vkb.landmarks import _cache_path
            p = _cache_path(vid, 5)
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

