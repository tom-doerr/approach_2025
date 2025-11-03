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


def test_tie_keeps_new_value_and_advances():
    ds = _DS()
    args = NS(brightness=0.5, warp=0.4, sat=0.3, contrast=0.2, hue=0.1, wb=0.05)
    dyn = AugScheduler(ds, args, patience=1)
    # First stall: raise brightness one step
    dyn.update(False)
    assert ds.brightness > 0.0 and dyn.idx == 0
    # Next epoch tie: keep brightness and advance pointer
    dyn.update(False, equal=True)
    assert ds.brightness > 0.0 and dyn.idx == 1
