import types


def test_compute_class_weights_balanced():
    from vkb.finetune import _compute_class_weights
    # two classes, 3 vs 1 counts
    samples = [("v", 0, 0), ("v", 1, 0), ("v", 2, 0), ("v", 3, 1)]
    tr_idx = [0,1,2,3]
    w = _compute_class_weights(samples, tr_idx, 2)
    # N/(C*nc) → 4/(2*3)=0.666.. and 4/(2*1)=2.0 → ratio ~3x
    assert w[1] > w[0] * 2.5

