import sys


def test_cli_parses_no_persistent_and_val_eq_eps(monkeypatch):
    from train_frames import parse_args
    argv = [
        'train_frames.py',
        '--clf','dl','--data','data_small',
        '--no-persistent-workers',
        '--val-eq-eps','0.005',
    ]
    monkeypatch.setenv('PYTEST_CURRENT_TEST','1')
    monkeypatch.setenv('VKB_VISDOM_DISABLE','1')
    monkeypatch.setenv('VKB_MLFLOW_DISABLE','1')
    monkeypatch.setenv('VKB_MODELS_DIR','/tmp/vkb_models_test_cli')
    monkeypatch.setattr(sys, 'argv', argv)
    args = parse_args()
    assert getattr(args, 'no_persistent_workers', False) is True
    assert abs(float(getattr(args, 'val_eq_eps', 0.0)) - 0.005) < 1e-9

