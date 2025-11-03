from types import SimpleNamespace


class _StubTele:
    def __init__(self):
        self.mlflow = None
    def metric(self, *a, **k):
        return None
    def scalar(self, *a, **k):
        return None
    def scalar2(self, *a, **k):
        return None
    def text(self, *a, **k):
        return None


class _CapConsole:
    def __init__(self):
        self.lines = []
    def print(self, *a, **k):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)


def test_diag_system_prints_sys_line():
    import importlib
    ft = importlib.import_module('vkb.finetune')
    cons = _CapConsole()
    tel = _StubTele()
    # Args with diag_system enabled and minimal fields
    args = SimpleNamespace(epochs=1, aug='none', brightness=0.0, warp=0.0, drop_path=0.0, dropout=0.0, diag_system=True)
    # Call the hook
    ft._epoch_after(cons, tel, 1, args, tr_acc=0.5, val_acc=0.5)
    # Look for the Sys: line
    assert any('Sys:' in line for line in cons.lines)

