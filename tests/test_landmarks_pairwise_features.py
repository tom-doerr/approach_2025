import numpy as np


def test_pairwise_distance_features_shape_and_norm():
    from vkb.landmarks import pairwise_distance_features
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(21, 2))
    v = pairwise_distance_features(pts)
    # 21 choose 2 = 210
    assert v.shape == (210,)
    s = float(v.sum())
    assert 0.99 < s < 1.01

