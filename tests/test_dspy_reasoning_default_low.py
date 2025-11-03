import sys
from train_frames import parse_args


def test_reasoning_effort_defaults_low(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    monkeypatch.setattr(sys, 'argv', ['prog', '--dspy-aug', '--dspy-openrouter'])
    args = parse_args()
    assert getattr(args, 'dspy_reasoning_effort', None) == 'low'
