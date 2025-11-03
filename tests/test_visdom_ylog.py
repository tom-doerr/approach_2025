from vkb.telemetry import Telemetry


class _Viz:
    def __init__(self):
        self.calls = []
    def line(self, **kwargs):
        self.calls.append(("line", kwargs))


def test_aug_strengths_uses_log_yaxis():
    viz = _Viz()
    tel = Telemetry(console=None, viz=viz, mlflow=None)
    tel.scalar2("aug_strengths", "brightness", 1, 0.1, title="Aug Strengths", ylabel="strength", win="vkb_policy_strengths")
    assert viz.calls, "expected a visdom line call"
    opts = viz.calls[0][1].get("opts", {})
    assert opts.get("ytype") == "log"
