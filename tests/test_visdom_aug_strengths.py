from vkb.telemetry import Telemetry


class _Viz:
    def __init__(self):
        self.calls = []
    def line(self, **kwargs):
        self.calls.append(("line", kwargs))


def test_scalar2_creates_and_appends_series():
    viz = _Viz()
    tel = Telemetry(console=None, viz=viz, mlflow=None)
    # first call creates window with legend [series]
    tel.scalar2("aug_strengths", "brightness", 1, 0.1, title="Aug Strengths", ylabel="strength", win="vkb_policy_strengths")
    # second call appends another series
    tel.scalar2("aug_strengths", "warp", 1, 0.2, title="Aug Strengths", ylabel="strength", win="vkb_policy_strengths")
    assert len(viz.calls) == 2
    assert viz.calls[0][1]['name'] == 'brightness'
    assert viz.calls[1][1]['name'] == 'warp'
