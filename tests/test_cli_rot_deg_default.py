def test_rot_deg_default_full_rotation(monkeypatch):
    import train_frames as tf
    # Simulate empty argv
    monkeypatch.setenv('PYTEST_CURRENT_TEST', '1')
    args = tf.parse_args([])
    assert float(getattr(args, 'rot_deg', 0.0)) == 360.0
