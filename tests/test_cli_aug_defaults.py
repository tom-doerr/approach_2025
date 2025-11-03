def test_aug_cli_defaults_stronger(monkeypatch):
    import train_frames as tf
    monkeypatch.setenv('PYTEST_CURRENT_TEST', '1')
    args = tf.parse_args([])
    assert args.brightness == 0.25
    assert args.sat == 0.40
    assert args.contrast == 0.15
    assert args.wb == 0.30
    assert args.hue == 0.30
    assert args.warp == 0.30
    assert args.noise_std == 0.05
