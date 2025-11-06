def test_xgb_expand_detects_edges():
    from train_frames import _needs_expand_xgb, _xgb_default_bounds
    b = _xgb_default_bounds()
    # Near upper n_estimators
    p = {'max_depth': b['depth_min']+1, 'n_estimators': b['n_estimators_max'], 'learning_rate': 10**((b['lr_exp_min']+b['lr_exp_max'])/2), 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 10**((b['reg_lambda_exp_min']+b['reg_lambda_exp_max'])/2)}
    assert _needs_expand_xgb(p, b)

def test_logreg_expand_detects_edges():
    from train_frames import _needs_expand_logreg
    assert _needs_expand_logreg(1e8, -4.0, 8.0)
    assert _needs_expand_logreg(1e-4, -4.0, 8.0)
