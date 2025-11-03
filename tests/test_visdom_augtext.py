import sys


class _FakeViz:
    def __init__(self):
        self.text_calls = []
    def text(self, txt, opts=None, win=None, append=False):
        self.text_calls.append({"txt": txt, "opts": opts, "win": win, "append": append})
        return win or "w_text"


def test_viz_aug_text_appends():
    from vkb.finetune import _viz_aug_text
    viz = _FakeViz(); wins = {}
    class A: pass
    a = A(); a.epochs=2; a.aug='rot360'; a.noise_std=0.1; a.brightness=0.2; a.warp=0.3; a.drop_path=0.25; a.dropout=0.0
    _viz_aug_text(viz, wins, a, 1)
    _viz_aug_text(viz, wins, a, 2)
    assert viz.text_calls[0]["append"] == False
    assert viz.text_calls[1]["append"] == True
