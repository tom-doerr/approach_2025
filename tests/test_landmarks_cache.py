import os, tempfile, shutil
import numpy as np


def test_landmarks_cache_skips_compute_on_second_call(monkeypatch):
    import vkb.landmarks as lm
    tmp = tempfile.mkdtemp()
    try:
        vid = os.path.join(tmp, 'x.mp4')
        open(vid, 'wb').close()  # empty file; we stub compute to avoid cv2

        calls = {'n': 0}

        def fake_compute(path, stride, max_frames):
            calls['n'] += 1
            idx = np.arange(0, 10, 2, dtype=int)
            lm_arr = np.zeros((len(idx), 21, 3), dtype=float)
            # encode unique signature for features
            for i in range(len(idx)):
                lm_arr[i, 0, 0] = float(i + 1)
            return idx, lm_arr

        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fake_compute)

        f1 = lm.extract_features_for_video(vid, stride=2, max_frames=5)
        assert calls['n'] == 1
        assert f1.shape[0] == 5
        # Second call with same stride should hit cache and not call compute
        def fail_compute(*a, **kw):
            raise AssertionError('compute called despite cache')
        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fail_compute)

        f2 = lm.extract_features_for_video(vid, stride=2, max_frames=5)
        assert f2.shape == f1.shape
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

