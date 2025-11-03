import io, sys, types


def test_policy_epoch_log_line(monkeypatch, tmp_path):
    import vkb.finetune as ft

    # Stubs
    monkeypatch.setattr(ft, '_prepare_data', lambda args, cons: ([], ['A','B'], [], 2, [0], [0], []))
    class DS:
        def __init__(self):
            self.brightness=0.11; self.warp=0.22; self.sat=0.33; self.contrast=0.44; self.hue=0.05; self.wb=0.06; self.rot_deg=12.0
            self._augment = type('Aug', (), {'warp': 0.0})()
    def _mk_loaders(args, *a, **k):
        args._ds_tr = DS(); return None, None, None, (False, 0, 0, False, 'auto')
    monkeypatch.setattr(ft, '_make_loaders', _mk_loaders)

    class M:
        def state_dict(self): return {}
    monkeypatch.setattr(ft, '_init_model_and_optim', lambda *a, **k: (M(), object(), object()))
    # return 7-tuple to include train_loss
    monkeypatch.setattr(ft, '_train_epoch', lambda *a, **k: (0.5, 10, [], [], [], 0.01, 0.3))
    monkeypatch.setattr(ft, '_validate_epoch', lambda *a, **k: (0.4, 0.7))

    import os
    monkeypatch.setenv('VKB_MODELS_DIR', str(tmp_path))
    args = ft.FTArgs(data='.', embed_model='stub', eval_split=0.0, eval_mode='tail', epochs=1, batch_size=2, lr=1e-4, wd=1e-4, device='cpu')
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        ft.finetune(args)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    assert 'Policy used:' in out
    assert 'train: loss=0.300 acc=0.500' in out
    assert 'val: loss=0.700 acc=0.400' in out
    assert 'brightness=' in out and 'warp=' in out
