def test_shift_cli_default_heavy(monkeypatch):
    import train_frames as tf
    args = tf.parse_args([])
    assert float(getattr(args, 'shift', 0.0)) == 0.25
