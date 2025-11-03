def test_cache_helpers_exist():
    import vkb.cache as vc
    for name in ("_src_fingerprint","_check_existing","_recheck_after_wait","_count_frames_dims","_build_frames_cache"):
        assert hasattr(vc, name), f"missing helper: {name}"

