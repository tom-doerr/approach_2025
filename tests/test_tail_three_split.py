def test_tail_split_three_classic():
    import train_frames as tf
    # two classes, 10 frames each
    idx_by_class = {0: list(range(10)), 1: list(range(10, 20))}
    tr, va, te = tf._split_tail_indices_three(idx_by_class, val_frac=0.1, test_frac=0.1)
    assert len(tr) == 16 and len(va) == 2 and len(te) == 2
    # disjoint and full cover
    all_idx = set(tr) | set(va) | set(te)
    assert all_idx == set(range(20)) and (set(tr) & set(va) & set(te)) == set()


def test_tail_split_three_dl():
    from vkb.finetune import _tail_split_three
    # Build samples: (video, frame, class)
    samples = []
    for i in range(10): samples.append(("A", i, 0))
    for i in range(10): samples.append(("B", i, 1))
    tr, va, te = _tail_split_three(samples, n_classes=2, val_frac=0.1, test_frac=0.1)
    assert len(tr) == 16 and len(va) == 2 and len(te) == 2
    # sanity: indices refer to original list
    assert max(tr + va + te) < len(samples)

