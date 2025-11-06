import os, tempfile, shutil
import numpy as np


def test_extract_features_unlimited_when_zero(monkeypatch):
    import vkb.landmarks as lm
    tmp = tempfile.mkdtemp()
    try:
        vid = os.path.join(tmp, 'x.mp4')
        open(vid, 'wb').close()

        def fake_compute(path, stride, max_frames):
            assert max_frames == 0  # pass-through of zero
            idx = np.arange(10, dtype=int)
            pts = np.zeros((10, 21, 3), dtype=float)
            # avoid degenerate zero-sum: put a small gradient on x
            for i in range(10):
                pts[i, 0, 0] = i + 1
            return idx, pts

        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fake_compute)
        feats = lm.extract_features_for_video(vid, stride=1, max_frames=0)
        assert feats.shape[0] == 10
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

