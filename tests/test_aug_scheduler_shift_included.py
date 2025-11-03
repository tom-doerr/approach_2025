def test_aug_scheduler_includes_shift_and_raises(monkeypatch):
    from vkb.aug_scheduler import AugScheduler
    import types

    ds = types.SimpleNamespace(shift=0.0)
    # Limit scheduler to only shift to keep it deterministic here
    sch = AugScheduler(ds, args=None, methods=["shift"], patience=1, steps=5)
    sch.p_decrease = 0.0  # force raise
    assert hasattr(ds, 'shift') and ds.shift == 0.0
    # first call: no improvement → raise shift one level
    sch.update(improved=False)
    assert ds.shift > 0.0

