import os, tempfile
import numpy as np


def test_cache_embeddings_roundtrip(monkeypatch):
    import vkb.cache as vc
    # isolate cache under a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(vc, 'cache_root', lambda: os.path.join(tmp, '.cache', 'vkb'))
        c = vc.Cache(model_name='m', data_root='data')
        vid = os.path.join('data', 'lab', 'v1.mp4')
        arr = np.arange(6, dtype=np.float32).reshape(2,3)
        p = c.save(vid, arr)
        assert os.path.exists(p)
        out = c.load(vid)
        assert out.shape == arr.shape and np.allclose(out, arr)


def test_cache_frames_calls_under_class(monkeypatch):
    import vkb.cache as vc
    called = {}
    monkeypatch.setattr(vc, 'ensure_frames_cached', lambda data_root, video_path, console=None: ( {'n':1,'h':8,'w':8}, True))
    mm = np.zeros((1,8,8,3), dtype=np.uint8)
    monkeypatch.setattr(vc, 'open_frames_memmap', lambda data_root, video_path: (mm, {'n':1,'h':8,'w':8}))
    c = vc.Cache(model_name='m', data_root='data')
    meta, built = c.ensure_frames('data/a.mp4')
    assert built is True and meta['n'] == 1
    mm2, meta2 = c.open_frames('data/a.mp4')
    assert mm2.shape == (1,8,8,3) and meta2['w'] == 8

