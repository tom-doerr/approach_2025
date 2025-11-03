import os
from vkb.vis import visdom_prepare


class _FakeViz:
    def __init__(self):
        self.closed = []
    def close(self, win):
        self.closed.append(win)


def test_visdom_prepare_does_not_clear_under_env(monkeypatch):
    viz = _FakeViz()
    monkeypatch.setenv('VKB_VISDOM_NO_CLEAR', '1')
    visdom_prepare(viz)
    assert viz.closed == []

