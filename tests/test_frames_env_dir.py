def test_frames_env_dir(monkeypatch, tmp_path):
    from vkb.cache import frames_paths
    # default path under .cache/vkb/frames
    meta0, data0 = frames_paths('data', '/abs/path/video.mp4')
    assert '.cache' in meta0 and meta0.endswith('.meta.json')
    # override via env
    monkeypatch.setenv('VKB_FRAMES_DIR', str(tmp_path))
    meta1, data1 = frames_paths('data', '/abs/path/video.mp4')
    assert str(tmp_path) in meta1
    assert meta1.endswith('.meta.json') and data1.endswith('.uint8')
