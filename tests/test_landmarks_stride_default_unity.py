import os, tempfile, shutil
import numpy as np


def test_landmarks_default_stride_is_one(monkeypatch):
    import vkb.landmarks as lm
    tmp = tempfile.mkdtemp()
    try:
        vid = os.path.join(tmp, 'x.mp4'); open(vid, 'wb').close()
        seen = {'stride': None}
        def fake_compute(path, stride, max_frames, start_from=0):
            seen['stride'] = stride
            idx = np.arange(3, dtype=int)
            pts = np.zeros((3,21,3), dtype=float); pts[:,0,0] = np.arange(1,4)
            return idx, pts
        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fake_compute)
        _ = lm.extract_features_for_video(vid)
        assert seen['stride'] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
