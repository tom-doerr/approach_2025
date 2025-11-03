def test_finetuner_classmethod_calls_fit(monkeypatch):
    import types
    import vkb.finetune as vf
    called = {}
    # Make fit return a sentinel via monkeypatching the global function used by fit
    orig = vf.finetune
    try:
        vf.finetune = lambda args: 'cm_ok'
        out = vf.Finetuner.finetune(types.SimpleNamespace())
        assert out == 'cm_ok'
    finally:
        vf.finetune = orig

