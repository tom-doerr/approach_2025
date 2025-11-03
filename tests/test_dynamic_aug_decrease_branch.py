from types import SimpleNamespace as NS
from vkb.finetune import AugScheduler


class _DS:
    def __init__(self):
        self.brightness = 0.5
        self.warp = 0.0
        self.sat = 0.0
        self.contrast = 0.0
        self.hue = 0.0
        self.wb = 0.0


def test_dynamic_aug_can_try_decrease_then_restore(monkeypatch):
    ds = _DS()
    args = NS(brightness=0.5, warp=0.4, sat=0.3, contrast=0.2, hue=0.1, wb=0.05)
    dyn = AugScheduler(ds, args, patience=1)
    # Set nonzero current after scheduler zeroed strengths
    ds.brightness = 0.5
    # Force pointer to brightness with nonzero current
    dyn.idx = 0
    # Force random to choose decrease path
    monkeypatch.setattr(__import__('random'), 'random', lambda: 0.1)
    # Stall → since current index is 0, we raise one step above zero
    dyn.update(False)
    assert ds.brightness > 0.0
    # Next epoch worse → revert to previous (0.5)
    dyn.update(False, equal=False)
    assert ds.brightness == 0.5
