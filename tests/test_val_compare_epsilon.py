from vkb.finetune import _val_compare, VAL_EQ_EPS


def test_val_compare_improved_over_eps():
    improved, equal = _val_compare(0.401, 0.398, eps=0.002)
    assert improved is True and equal is False


def test_val_compare_tie_within_eps():
    # within epsilon → treat as equal (keep aug change)
    improved, equal = _val_compare(0.4005, 0.4000, eps=0.001)
    assert improved is False and equal is True


def test_val_compare_worse_below_eps():
    improved, equal = _val_compare(0.395, 0.400, eps=0.002)
    assert improved is False and equal is False


def test_val_compare_none_best_is_improvement():
    improved, equal = _val_compare(0.1, None, eps=VAL_EQ_EPS)
    assert improved is True and equal is False

