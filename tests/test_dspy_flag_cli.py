import sys
from train_frames import parse_args


def test_flag_parses_and_defaults_off(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    old = sys.argv[:]
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setattr(sys, 'argv', ['prog'])
    args = parse_args()
    assert getattr(args, 'dspy_aug', False) is False


def test_flag_parses_on(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    import sys as _sys
    monkeypatch.setattr(_sys, 'argv', ['prog', '--dspy-aug', '--dspy-openrouter', '--dspy-reasoning-effort', 'high', '--dspy-stream-reasoning', '--dspy-model', 'deepseek/deepseek-reasoner'])
    args = parse_args()
    assert getattr(args, 'dspy_aug', False) is True
    assert getattr(args, 'dspy_openrouter', False) is True
    assert getattr(args, 'dspy_reasoning_effort', None) == 'high'
    assert getattr(args, 'dspy_stream_reasoning', False) is True
    assert getattr(args, 'dspy_model', '') == 'deepseek/deepseek-reasoner'
