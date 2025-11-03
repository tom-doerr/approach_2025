def test_viz_scalar_appends(monkeypatch):
    calls = []
    class FakeViz:
        def line(self, X=None, Y=None, opts=None, win=None, update=None, name=None):
            calls.append({"X": X, "Y": Y, "win": win, "update": update, "title": (opts or {}).get("title")})
            return "w1" if win is None else win

    from vkb.finetune import _viz_scalar
    viz = FakeViz()
    wins = {}
    _viz_scalar(viz, wins, "train", 1, 0.9, "Train/Val Acc")
    assert 'acc' in wins and calls[-1]["win"] == 'vkb_acc'
    _viz_scalar(viz, wins, "train", 2, 0.92, "Train/Val Acc")
    assert calls[-1]["update"] == "append"
