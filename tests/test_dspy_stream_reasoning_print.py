import io, sys, types


def test_stream_reasoning_invoked(monkeypatch, tmp_path):
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

    # Stub dspy agents: predictor instant; stream pushes deltas immediately
    seen = []
    def _stream_reasoning_openrouter(steps, model, api_key=None, effort=None, on_delta=None, api_base=None):
        if on_delta:
            on_delta("alpha"); seen.append('alpha')
            on_delta("beta"); seen.append('beta')

    mod = types.SimpleNamespace(
        AsyncAugPredictor=type('P', (), {
            '__init__': lambda self: None,
            'submit': lambda self, hist: None,
            'ready': lambda self: True,
            'result': lambda self, timeout=0.0: {
                'brightness': 0.12, 'warp': 0.22, 'sat': 0.05, 'contrast': 0.1, 'hue': 0.01, 'wb': 0.02, 'rot_deg': 45.0,
            }
        }),
        configure_deepseek=lambda *a, **k: None,
        configure_openrouter=lambda *a, **k: None,
        stream_reasoning_openrouter=_stream_reasoning_openrouter,
        AugStep=type('AugStep', (), {}),
    )
    sys.modules['dspy_agents'] = mod

    monkeypatch.setenv('VKB_MODELS_DIR', str(tmp_path))
    args = ft.FTArgs(data='.', embed_model='stub', eval_split=0.0, eval_mode='tail', epochs=1, batch_size=2, lr=1e-4, wd=1e-4, device='cpu')
    setattr(args, 'dspy_aug', True)
    setattr(args, 'dspy_openrouter', True)
    setattr(args, 'dspy_stream_reasoning', True)
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        ft.finetune(args)
        import time as _t
        _t.sleep(0.05)
    finally:
        sys.stdout = old
        sys.modules.pop('dspy_agents', None)
    assert seen, "expected stream_reasoning_openrouter to be invoked"
