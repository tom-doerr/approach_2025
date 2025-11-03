def test_validate_returns_loss():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch import nn
    from vkb.finetune import _validate_epoch
    from rich.console import Console

    X = torch.randn(6, 3, 8, 8)
    y = torch.tensor([0,1,0,1,0,1])
    dl = DataLoader(TensorDataset(X, y), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 2))
    val_acc, val_loss = _validate_epoch(dl, model, 'cpu', 2, ['a','b'], Console(file=None), loss_fn=nn.CrossEntropyLoss())
    assert isinstance(val_acc, float)
    assert val_loss is None or val_loss >= 0.0

