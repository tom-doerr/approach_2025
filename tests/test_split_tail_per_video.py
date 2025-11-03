import vkb.finetune as ft
import train_frames as tf


def test_dl_per_video_tail_three_basic():
    # Two videos, sizes 5 and 4
    samples = []
    v1, v2 = 'v1.mp4', 'v2.mp4'
    # video 1: 5 frames, class 0 → indices 0..4
    for i in range(5):
        samples.append((v1, i, 0))
    # video 2: 4 frames, class 1 → indices 5..8
    for i in range(4):
        samples.append((v2, i, 1))
    tr, va, te = ft._per_video_tail_split_three(samples, val_frac=0.4, test_frac=0.2)
    # v1: c=5 → vv=2, tv=1, head=2 → tr [0,1], va [2,3], te [4]
    # v2: c=4 → vv=2, tv=1, head=1 → tr [5], va [6,7], te [8]
    assert tr == [0, 1, 5]
    assert va == [2, 3, 6, 7]
    assert te == [4, 8]


def test_classic_split_tail_per_video_slices():
    # Two slices: [0..5) and [5..9)
    slices = [(0, 5), (5, 9)]
    tr, va, te = tf._split_tail_per_video_slices(slices, val_frac=0.4, test_frac=0.0)
    # Video 1: c=5 → vv=2, head=3 → train 0,1,2; val 3,4
    # Video 2: c=4 → vv=2, head=2 → train 5,6; val 7,8
    assert tr == [0, 1, 2, 5, 6]
    assert va == [3, 4, 7, 8]
    assert te == []
