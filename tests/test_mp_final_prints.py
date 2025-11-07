import io, os, tempfile


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_logreg_prints_final_params_and_val(monkeypatch):
    import train_frames as tf
    tmp = tempfile.mkdtemp()
    _mk_dataset(tmp)
    # Stub features: two classes, 6 frames each
    import vkb.landmarks as lm, numpy as np
    monkeypatch.setattr(lm, 'extract_features_for_video', lambda *a, **k: np.ones((6,210)))
    # Capture console
    from rich.console import Console as RealConsole
    buf = io.StringIO(); monkeypatch.setattr('rich.console.Console', lambda *a, **k: RealConsole(file=buf, force_terminal=False, width=120))
    args = tf.parse_args(['--data', tmp, '--clf', 'mp_logreg', '--eval-split', '0.5', '--C', '1.0'])
    tf.train(args)
    out = buf.getvalue()
    assert 'Saved model:' in out
    assert 'Final: val_acc=' in out and 'params=C=' in out


def test_mp_xgb_prints_final_params_and_val(monkeypatch):
    import train_frames as tf
    tmp = tempfile.mkdtemp()
    _mk_dataset(tmp)
    # Stub features
    import vkb.landmarks as lm, numpy as np
    monkeypatch.setattr(lm, 'extract_features_for_video', lambda *a, **k: np.ones((6,210)))
    # Stub xgboost to avoid heavy import
    class StubXGB:
        def __init__(self, **params): self.params = params
        def fit(self, X, y): return self
        def score(self, X, y): return 1.0
        def get_params(self, deep=True): return self.params
    import sys, types
    old_xgb = sys.modules.get('xgboost')
    sys.modules['xgboost'] = types.SimpleNamespace(XGBClassifier=StubXGB)
    try:
        from rich.console import Console as RealConsole
        buf = io.StringIO(); monkeypatch.setattr('rich.console.Console', lambda *a, **k: RealConsole(file=buf, force_terminal=False, width=120))
        # Avoid pickling local StubXGB
        import vkb.artifacts as art
        old_save = art.save_model
        monkeypatch.setattr(art, 'save_model', lambda *a, **k: 'models/dummy.pkl')
        args = tf.parse_args(['--data', tmp, '--clf', 'mp_xgb', '--eval-split', '0.5', '--hpo-xgb', '0'])
        tf.train(args)
        out = buf.getvalue()
        assert 'Saved model:' in out
        assert 'Final: val_acc=' in out and 'params=' in out
    finally:
        try:
            art.save_model = old_save  # type: ignore
        except Exception:
            pass
        if old_xgb is not None: sys.modules['xgboost'] = old_xgb
        else: del sys.modules['xgboost']
