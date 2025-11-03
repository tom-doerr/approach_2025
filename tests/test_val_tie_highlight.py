import io
import sys
import types


def test_val_tie_highlight_prints(monkeypatch):
    import vkb.finetune as ft

    # Stub timm to avoid heavy import
    import types, sys as _sys
    _sys.modules['timm'] = types.SimpleNamespace(create_model=lambda *a, **k: None)

    # Stub minimal pipeline pieces
    monkeypatch.setattr(ft, '_prepare_data', lambda args, cons: ([], ['A','B'], [], 2, [0], [0], []))
    monkeypatch.setattr(ft, '_make_loaders', lambda args, samples, tr, va, n, device, te_idx=None: (None, None, None, (False, 0, 0, False, 'auto')))
    class _M: 
        def state_dict(self):
            return {}
    monkeypatch.setattr(ft, '_init_model_and_optim', lambda *a, **k: (_M(), object(), object()))
    # Train epoch stub
    monkeypatch.setattr(ft, '_train_epoch', lambda *a, **k: (0.5, 10, [], [], [], 0.01))
    # Validation returns best then tie within eps
    vals = iter([0.400, 0.409])
    monkeypatch.setattr(ft, '_validate_epoch', lambda *a, **k: next(vals))

    args = ft.FTArgs(data='.', embed_model='stub', eval_split=0.0, eval_mode='tail', epochs=2, batch_size=2, lr=1e-4, wd=1e-4, device='cpu')
    # Make epsilon larger than delta (0.009) so it's a tie
    setattr(args, 'val_eq_eps', 0.01)
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        ft.finetune(args)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    assert 'Tie within ±0.010' in out
