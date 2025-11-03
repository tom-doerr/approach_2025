def test_cli_default_erase_enabled(monkeypatch):
    import train_frames as tf
    monkeypatch.setenv('PYTEST_CURRENT_TEST', '1')
    args = tf.parse_args([])
    assert 0.0 < float(getattr(args, 'erase_p', 0.0)) <= 1.0
