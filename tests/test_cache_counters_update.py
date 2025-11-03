def test_cache_counters_update(monkeypatch):
    import vkb.dataset as d
    # Stub cache to simulate one build then reuse
    calls = {"n": 0}
    def _ensure(data_root, vp):
        calls["n"] += 1
        # built True on first call, False afterwards
        return ({"n": 1, "h": 2, "w": 2}, calls["n"] == 1)
    def _open(data_root, vp):
        import numpy as np
        return (np.zeros((1,2,2,3), dtype=np.uint8), {"n": 1, "h": 2, "w": 2})
    monkeypatch.setattr('vkb.cache.ensure_frames_cached', _ensure)
    monkeypatch.setattr('vkb.cache.open_frames_memmap', _open)

    ds = d.FrameDataset([("vidA", 0, 0), ("vidA", 0, 0)], img_size=4, data_root='.', mode='train')
    _ = ds[0]
    _ = ds[1]
    assert ds._cache_misses == 1 and ds._cache_hits == 0 and len(ds._seen_videos) == 1

