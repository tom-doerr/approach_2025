def test_finetuner_fit_aliases_run():
    import types
    import vkb.finetune as vf
    called = {}
    orig = vf.finetune
    try:
        vf.finetune = lambda a: 'ok'
        ft = vf.Finetuner(types.SimpleNamespace())
        assert ft.fit() == 'ok'
        assert ft.run() == 'ok'
    finally:
        vf.finetune = orig

