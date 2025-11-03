class _FakeViz:
    def __init__(self):
        self.calls = []
    def images(self, X=None, nrow=8, opts=None, win=None):
        self.calls.append({"win": win, "nrow": nrow, "opts": opts})
        return win or "w_aug"


def test_viz_images_reuses_window():
    from vkb.finetune import _train_epoch
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch import nn
    X = torch.rand(4,3,224,224); y = torch.tensor([0,1,0,1])
    dl = DataLoader(TensorDataset(X,y), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3*224*224, 2))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    class A: pass
    args = A(); args.epochs=1; args.visdom_aug=4
    viz = _FakeViz(); wins = {}
    from rich.console import Console
    _train_epoch(1, args, dl, model, 'cpu', opt, loss_fn, Console(file=None), prog=None, viz=viz, viz_wins=wins)
    # First call uses fixed window id
    assert viz.calls[0]['win'] == 'vkb_aug'
    # Stored id
    assert wins.get('aug_images') == 'vkb_aug'
    # Simulate a second log and assert update uses same window
    viz.calls.clear()
    _train_epoch(1, args, dl, model, 'cpu', opt, loss_fn, Console(file=None), prog=None, viz=viz, viz_wins=wins)
    assert viz.calls and viz.calls[0]['win'] == 'vkb_aug'
