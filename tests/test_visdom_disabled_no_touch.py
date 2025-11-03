from types import SimpleNamespace as NS

from vkb.vis import setup_visdom


class _C:
    def __init__(self):
        self.lines = []
    def print(self, msg):
        self.lines.append(str(msg))


def test_setup_visdom_returns_none_and_silent_when_features_off():
    args = NS(visdom_aug=0, visdom_metrics=False)
    cons = _C()
    viz = setup_visdom(args, cons)
    assert viz is None
    assert not any('Visdom' in s for s in cons.lines)

