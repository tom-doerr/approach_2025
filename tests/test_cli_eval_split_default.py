import argparse
import train_frames as tf


def test_cli_eval_split_default_is_one_percent():
    p = tf.parse_args([])
    assert abs(float(p.eval_split) - 0.01) < 1e-9

