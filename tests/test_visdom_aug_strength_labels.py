from rich.console import Console


class _FakeViz:
    def __init__(self):
        self.lines = []
    def line(self, X=None, Y=None, win=None, name=None, opts=None, update=None):
        self.lines.append({"win": win, "name": name, "opts": opts, "update": update, "X": X, "Y": Y})


def test_visdom_aug_strength_explicit_labels(monkeypatch):
    import vkb.finetune as ft
    from vkb.telemetry import Telemetry

    class DS:
        brightness=0.1; warp=0.2; sat=0.3; contrast=0.4; hue=0.05; wb=0.06; rot_deg=12.0

    class A: pass
    args = A(); args.dynamic_aug=True; args._ds_tr=DS(); args.epochs=2; args.dropout=0.1; args.drop_path=0.2
    viz = _FakeViz(); telemetry = Telemetry(Console(file=None), viz=viz, mlflow=None)

    # call epoch_after to emit aug strengths
    ft._epoch_after(Console(file=None), telemetry, 1, args, tr_acc=0.5, val_acc=0.4)

    names = {rec["name"] for rec in viz.lines if rec["win"] == 'vkb_policy_strengths'}
    # Title should be renamed to Policy (Aug + Reg)
    first = next((rec for rec in viz.lines if rec["win"] == 'vkb_policy_strengths' and rec["opts"]), None)
    assert first and first["opts"]["title"] == "Policy (Aug + Reg)"
    # Expect explicit labels
    for label in ("brightness","warp","saturation","contrast","hue","white_balance","rotation_deg","dropout","drop_path"):
        assert label in names
