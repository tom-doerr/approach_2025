import io, sys


def test_epoch1_cache_summary_probes_caches(monkeypatch):
    # We call _epoch_after directly and verify it prints a Cache: line that
    # reflects ensure_frames_cached results, independent of worker counters.
    from vkb import finetune as ft

    # Stub ensure_frames_cached: two videos, both hits (built=False)
    import vkb.cache as vc
    calls = []
    def ens(data_root, vp):
        calls.append(vp)
        return {"n": 2, "h": 4, "w": 5}, False
    monkeypatch.setattr(vc, 'ensure_frames_cached', ens)

    # Minimal args
    from types import SimpleNamespace
    args = SimpleNamespace(
        data='.', epochs=1, aug='none', brightness=0.0, warp=0.0,
        drop_path=0.0, dropout=0.0, _vids=[('v1.mp4','A'), ('v2.mp4','A')]
    )

    class Cons:
        def __init__(self): self.buf = io.StringIO()
        def print(self, *a, **k):
            txt = " ".join(str(x) for x in a)
            self.buf.write(txt + "\n")

    cons = Cons()
    # No telemetry for this unit
    ft._epoch_after(cons, telemetry=type('T',(),{'metric':lambda *a,**k: None, 'scalar':lambda *a,**k: None, 'text': lambda *a,**k: None, 'scalar2': lambda *a,**k: None})(), ep=1, args=args, tr_acc=0.5, val_acc=0.4)
    out = cons.buf.getvalue()
    assert 'Cache: videos=2' in out and 'hits=' in out and 'misses=' in out
    # ensure we actually probed both paths
    assert calls == ['v1.mp4', 'v2.mp4']

