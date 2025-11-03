import io, sys, types


def test_no_print_when_not_ready(monkeypatch, tmp_path):
    import vkb.finetune as ft

    # Minimal stubs for data + loaders
    monkeypatch.setattr(ft, '_prepare_data', lambda args, cons: ([], ['A','B'], [], 2, [0], [0], []))

    class DS:
        def __init__(self):
            self.brightness=0.0; self.warp=0.0; self.sat=0.0; self.contrast=0.0; self.hue=0.0; self.wb=0.0; self.rot_deg=0.0
            self._augment = type('Aug', (), {'warp': 0.0})()

    def _mk_loaders(args, *a, **k):
        args._ds_tr = DS()
        return None, None, None, (False, 0, 0, False, 'auto')

    monkeypatch.setattr(ft, '_make_loaders', _mk_loaders)

    class _M:
        def state_dict(self):
            return {}
    monkeypatch.setattr(ft, '_init_model_and_optim', lambda *a, **k: (_M(), object(), object()))
    monkeypatch.setattr(ft, '_train_epoch', lambda *a, **k: (0.5, 10, [], [], [], 0.01))
    monkeypatch.setattr(ft, '_validate_epoch', lambda *a, **k: (0.4, 0.7))

    # Stub predictor never ready
    mod = types.SimpleNamespace(
        AsyncAugPredictor=type('P', (), {
            '__init__': lambda self: None,
            'submit': lambda self, hist: None,
            'ready': lambda self: False,
            'result': lambda self, timeout=0.0: (_ for _ in ()).throw(AssertionError('should not be called')),
        }),
        configure_deepseek=lambda *a, **k: None,
        configure_openrouter=lambda *a, **k: None,
    )
    sys.modules['dspy_agents'] = mod

    monkeypatch.setenv('VKB_MODELS_DIR', str(tmp_path))
    args = ft.FTArgs(data='.', embed_model='stub', eval_split=0.0, eval_mode='tail', epochs=1, batch_size=2, lr=1e-4, wd=1e-4, device='cpu')
    setattr(args, 'dspy_aug', True)
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        ft.finetune(args)
    finally:
        sys.stdout = old
        sys.modules.pop('dspy_agents', None)
    out = buf.getvalue()
    assert 'DSPy aug applied for next epoch' not in out
