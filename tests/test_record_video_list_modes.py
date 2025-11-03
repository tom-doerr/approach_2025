import io, sys


def test_list_modes_runs_v4l2ctl(monkeypatch):
    import record_video as rv

    # Pretend we're on Linux with v4l2-ctl available
    monkeypatch.setattr(rv, 'os', __import__('os'))  # no-op
    import shutil as _sh
    monkeypatch.setattr(_sh, 'which', lambda cmd: '/usr/bin/v4l2-ctl' if cmd=='v4l2-ctl' else None)

    class R:
        def __init__(self):
            self.returncode = 0
            self.stdout = 'Fake formats list\n\tSize: Discrete 640x480, 30 fps\n'
            self.stderr = ''

    import subprocess as _sp
    monkeypatch.setattr(_sp, 'run', lambda *a, **k: R())

    # Call internal helper and assert output contains our fake line
    txt = rv._list_modes_linux(0)
    assert 'Fake formats list' in txt

