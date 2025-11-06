import os, tempfile, shutil
import numpy as np


def test_landmarks_memmap_build_and_read(monkeypatch):
    import vkb.landmarks as lm
    tmp = tempfile.mkdtemp()
    try:
        vid = os.path.join(tmp, 'x.mp4'); open(vid, 'wb').close()
        # Stub compute: detect hand at frames 0,2,4
        def fake_compute(path, stride, max_frames, start_from=0):
            idx = np.array([i for i in range(start_from, start_from+10) if i % 2 == 0], dtype=int)
            pts = np.zeros((len(idx), 21, 3), dtype=float)
            pts[:,0,0] = np.arange(1, len(idx)+1)
            return idx, pts
        monkeypatch.setattr(lm, '_compute_landmarks_for_stride', fake_compute)
        # Stub frame counter to 10 frames
        import vkb.cache as vc
        monkeypatch.setattr(vc, '_count_frames_dims', lambda p: (10, 8, 8))

        # Build memmap implicitly via extractor
        X = lm.extract_features_for_video(vid, stride=2, max_frames=0)
        # Should keep frames 0,2,4,6,8 (5 samples)
        assert X.shape[0] == 5
        # Fewer with stride=4
        X2 = lm.extract_features_for_video(vid, stride=4, max_frames=0)
        assert X2.shape[0] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

