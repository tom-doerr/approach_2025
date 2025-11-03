import types


def test_hpo_xgb_allows_larger_estimators_and_lambda(monkeypatch):
    import train_frames as tf
    # Craft tiny dataset
    X = [[0.0],[1.0],[2.0],[3.0],[4.0],[5.0]]
    y = [0,1,0,1,0,1]
    idx = {0:[0,2,4], 1:[1,3,5]}

    # Force RNG to pick upper bounds for n_estimators and reg_lambda
    import numpy as np, sys, types
    real_default_rng = np.random.default_rng  # capture before monkeypatch
    class FakeRNG:
        def __init__(self):
            self._real = real_default_rng(0)
        def integers(self, low, high, size=None, dtype=None, endpoint=False):
            val = (high - 1) if not endpoint else high
            import numpy as _np
            return _np.full(size or (), val, dtype=dtype or int)
        def uniform(self, low=0.0, high=1.0, size=None):
            import numpy as _np
            # broadcast like numpy; ignore exact semantics, prefer upper bound
            if size is None:
                return high
            return _np.full(size, high)
        # delegate other methods used by sklearn/scipy
        def __getattr__(self, name):
            return getattr(self._real, name)

    orig = np.random.default_rng
    monkeypatch.setattr(np.random, 'default_rng', lambda seed=None: FakeRNG())

    # Patch xgboost with a lightweight stub
    class StubXGB:
        def __init__(self, **params):
            self.params = params
        def fit(self, X, y):
            return self
        def score(self, X, y):
            return 1.0
    old_xgb = sys.modules.get('xgboost')
    sys.modules['xgboost'] = types.SimpleNamespace(XGBClassifier=StubXGB)
    try:
        p, s, trials = tf._hpo_xgb(X, y, iters=1, seed=0, idx_by_class=idx, eval_frac=0.5)
    finally:
        monkeypatch.setattr(np.random, 'default_rng', orig)
        if old_xgb is not None:
            sys.modules['xgboost'] = old_xgb
        else:
            del sys.modules['xgboost']

    # Assert chosen params include n_estimators near new max and reg_lambda near new max
    assert p['n_estimators'] >= 900
    assert p['reg_lambda'] >= 50
