import types


def test_live_val_step_logs_scalar(monkeypatch):
    import torch
    import vkb.finetune as ft

    # Minimal args with a one-batch val loader that shuffles
    xb = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    yb = torch.tensor([1, 0], dtype=torch.long)

    class DL:
        def __iter__(self):
            # two batches then stop
            yield xb, yb

    args = types.SimpleNamespace(_dl_val_live=DL(), _dl_val_it=None, val_live_interval=1)

    # Tiny model that predicts y perfectly
    class M(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros((x.size(0), 2), dtype=torch.float32)
            out[0,1] = 10.0; out[1,0] = 10.0
            return out

    model = M()

    logged = []
    class Tele:
        def scalar2(self, win_key, series, step, value, title=None, ylabel=None, win=None):
            logged.append((series, float(value)))
        def metric(self, *a, **k):
            pass

    class Cons:
        def print(self, *a, **k):
            pass

    ft._live_val_step(args, model, 'cpu', Tele(), Cons(), ep=1, global_step=1)
    assert any(s == 'val_live' and abs(v - 1.0) < 1e-6 for s, v in logged)
