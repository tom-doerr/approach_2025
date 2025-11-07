import io, sys, tempfile, os


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_xgb_hpo_prints_params_and_acc(monkeypatch):
    import train_frames as tf
    tmp = tempfile.mkdtemp()
    try:
        _mk_dataset(tmp)
        # Stub features to avoid mediapipe
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', lambda *a, **k: __import__('numpy').ones((6,210)))

        # Force _hpo_xgb to call logger twice with known params and scores
        calls = []
        def fake_hpo(X, y, iters, seed=0, logger=None, **_):
            p1 = {'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_lambda': 0.1}
            p2 = {'max_depth': 7, 'n_estimators': 400, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_lambda': 1.0}
            if logger:
                logger(1, p1, 0.75)
                logger(2, p2, 0.80)
            return p2, 0.80, [(p1, 0.75), (p2, 0.80)]
        orig = tf._hpo_xgb; tf._hpo_xgb = fake_hpo

        # Capture console
        from rich.console import Console as RealConsole
        buf = io.StringIO()
        # Monkeypatch global Console constructor to capture output
        monkeypatch.setattr('rich.console.Console', lambda *a, **k: RealConsole(file=buf, force_terminal=False, width=120))

        args = tf.parse_args(['--data', tmp, '--clf', 'mp_xgb', '--eval-split', '0.5', '--hpo-xgb', '2'])
        tf.train(args)
        out = buf.getvalue()
        assert 'trial 1: val_acc=0.750 params=' in out
        assert 'trial 2: val_acc=0.800 params=' in out
    finally:
        tf._hpo_xgb = orig
