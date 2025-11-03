from types import SimpleNamespace as NS
from vkb.finetune import AugScheduler


class _DS:
    def __init__(self):
        self.brightness = 0.5
        self.warp = 0.4
        self.sat = 0.3
        self.contrast = 0.2
        self.hue = 0.1
        self.wb = 0.05


def test_dynamic_aug_round_robin_behavior():
    ds = _DS()
    args = NS(brightness=0.5, warp=0.4, sat=0.3, contrast=0.2, hue=0.1, wb=0.05)
    dyn = AugScheduler(ds, args, patience=1)
    # all zeroed on init
    assert ds.brightness == 0.0 and ds.warp == 0.0 and ds.sat == 0.0 and ds.contrast == 0.0 and ds.hue == 0.0 and ds.wb == 0.0
    # 1 no-improve → raise brightness one step above zero
    dyn.update(False)  # first stall raises brightness
    assert ds.brightness > 0.0 and ds.warp == 0.0
    # Next: simulate worse (not equal) → revert and move to warp
    dyn.update(False, equal=False)
    assert ds.brightness == 0.0 and dyn.idx == 1
    # 1 more no-improve → raise warp one step
    dyn.update(False)
    assert ds.warp > 0.0
    # Improvement → keep warp, advance pointer
    dyn.update(True)
    assert ds.warp > 0.0 and dyn.idx == 2
