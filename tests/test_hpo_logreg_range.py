import numpy as np


def test_hpo_logreg_range_includes_large_C():
    from train_frames import _hpo_logreg
    X = np.concatenate([np.zeros((20,3)), np.ones((20,3))], axis=0)
    y = np.array([0]*20 + [1]*20)
    idx = {0: list(range(0,20)), 1: list(range(20,40))}
    C, s, trials = _hpo_logreg(X, y, iters=30, seed=0, idx_by_class=idx, eval_frac=0.3)
    Cs = [Cv for Cv,_ in trials]
    assert min(Cs) <= 1e-3
    assert max(Cs) >= 1e3
