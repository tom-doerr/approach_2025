def test_aug_scheduler_can_raise_rotation():
    from vkb.aug_scheduler import AugScheduler
    class DS:
        # mimic training dataset with a rot_deg field used by FrameDataset
        rot_deg = 0.0
    ds = DS()
    sch = AugScheduler(ds, args=None, methods=["rot_deg"], patience=1, steps=3)
    # Force always-raise behavior
    sch.p_decrease = 0.0
    assert getattr(ds, 'rot_deg', 0.0) == 0.0
    # First call (no improvement) schedules a raise
    sch.update(improved=False, equal=False)
    # Second call (monitor) commits change on tie or better; mark as equal
    sch.update(improved=False, equal=True)
    assert getattr(ds, 'rot_deg', 0.0) > 0.0
