def test_make_sample_weights_balanced():
    from train_frames import _make_sample_weights_from_labels
    y = [0]*9 + [1]
    w = _make_sample_weights_from_labels(y, n_classes=2)
    # 10 samples, 2 classes: weights = N/(C*n_c)
    # class0: 10/(2*9) = 10/18; class1: 10/(2*1) = 5.0
    # Check relative scaling conservatively
    import numpy as np
    w = np.asarray(w)
    w0 = float(w[:9].mean())
    w1 = float(w[-1])
    assert w1 > w0 * 8.0  # ~9x, allow slack
    # Sum of weights per class should be similar
    s0 = float(w[:9].sum())
    s1 = float(w[-1])
    assert 0.8 <= (s0 / s1) <= 1.25
