import types


def test_hpo_uses_early_stopping(monkeypatch, tmp_path):
    import hpo_mp_xgb as hpo
    calls = {}

    class FakeClf:
        def __init__(self, **p):
            self.p = p
        def fit(self, X=None, y=None, **kwargs):
            calls['eval_set'] = kwargs.get('eval_set')
            # accept both styles silently
            return self
        def predict_proba(self, X):
            import numpy as np
            # 2-class dummy probs
            return np.tile([0.4, 0.6], (len(X), 1))

    monkeypatch.setattr('xgboost.XGBClassifier', FakeClf)
    # Tiny dataset structure
    d = tmp_path
    (d/'A').mkdir(); (d/'B').mkdir()
    (d/'A'/'a.mp4').write_bytes(b'')
    (d/'B'/'b.mp4').write_bytes(b'')
    res = hpo.run_hpo(data=str(d), eval_split=0.5, trials=1, seed=0, eval_mode='tail', mp_stride=1, mp_max_frames=0, use_stub=True, early_rounds=33)
    assert calls.get('eval_set') is not None
