import sys


class _FakeViz:
    def __init__(self, *a, **kw):
        self.calls = []
    def images(self, arr, nrow=8, opts=None, win=None):
        self.calls.append((arr, nrow, opts, win))


def test_train_epoch_visdom_logs(monkeypatch):
    # Inject fake visdom client
    fake_mod = type("_M", (), {"Visdom": _FakeViz})
    monkeypatch.setitem(sys.modules, 'visdom', fake_mod)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch import nn
    from vkb.finetune import _train_epoch
    from rich.console import Console

    X = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3*224*224, 2))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    class A: pass
    args = A(); args.epochs = 1; args.visdom_aug = 2
    cons = Console(file=None)  # quiet
    viz = _FakeViz()

    _train_epoch(1, args, dl, model, 'cpu', opt, loss_fn, cons, prog=None, viz=viz)
    assert viz.calls, "expected at least one visdom images() call"
    arr, nrow, opts, win = viz.calls[0]
    assert win == 'vkb_aug'
