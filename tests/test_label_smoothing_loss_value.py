def test_label_smoothing_loss_value(monkeypatch):
    import types
    import torch
    import vkb.finetune as ft

    # Stub model factory to avoid timm
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc = torch.nn.Linear(4, 2, bias=False)
        def parameters(self):
            return self.fc.parameters()
        def forward(self, x):
            return self.fc(x)
    monkeypatch.setattr(ft, '_create_timm_model', lambda *a, **k: M())

    class A: pass
    a = A(); a.embed_model='stub'; a.dropout=0.0; a.drop_path=0.0; a.lr=1e-4; a.wd=1e-4
    a.class_weights = 'none'; a.label_smoothing = 0.12
    model, opt, loss_fn = ft._init_model_and_optim(a, n_classes=2, device='cpu', cons=types.SimpleNamespace(print=lambda *a, **k: None), loader_cfg=(False,0,0,False,'auto'), samples=[], tr_idx=[])

    # PyTorch exposes label_smoothing on CrossEntropyLoss in recent versions
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    # Some builds store it as attribute; verify numeric equality if present
    ls = getattr(loss_fn, 'label_smoothing', None)
    assert ls is None or abs(ls - 0.12) < 1e-9

