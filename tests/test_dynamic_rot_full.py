from vkb.finetune import AugScheduler


class _DS:
    def __init__(self):
        self.rot_deg = 0.0


def test_dynamic_aug_can_reach_full_rotation():
    ds = _DS()
    # Only control rotation; make steps modest and deterministic (no decreases)
    sch = AugScheduler(ds, args=None, methods=["rot_deg"], patience=1, steps=10)
    sch.p_decrease = 0.0
    # Levels should end at 360 exactly
    assert abs(sch.levels["rot_deg"][-1] - 360.0) < 1e-6
    # Walk through all raise cycles until the last level
    for _ in range(sch.steps):
        sch.update(improved=False)              # trigger raise
        sch.update(improved=False, equal=True)  # accept and advance
    assert ds.rot_deg >= 360.0 - 1e-6

