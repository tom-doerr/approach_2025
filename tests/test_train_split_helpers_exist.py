def test_train_split_helpers_exist():
    import train_frames as tf
    assert callable(getattr(tf, '_train_dl', None))
    assert callable(getattr(tf, '_train_classic', None))
    assert callable(getattr(tf, '_prepare_classic_io', None))
    assert callable(getattr(tf, '_embed_videos', None))
    assert callable(getattr(tf, '_fit_classic_and_save', None))


def test_train_delegates_dl(monkeypatch):
    import types
    import train_frames as tf
    called = {'x': False}
    def _finetune(args):
        called['x'] = True
        return 'DL_OK'
    mod = types.SimpleNamespace(finetune=_finetune)
    monkeypatch.setitem(__import__('sys').modules, 'vkb.finetune', mod)
    class A: pass
    a = A(); a.clf='dl'
    out = tf.train(a)
    assert called['x'] and out == 'DL_OK'


def test_train_delegates_classic(monkeypatch):
    import train_frames as tf
    called = {'x': False}
    monkeypatch.setattr(tf, '_train_classic', lambda args: (called.__setitem__('x', True) or 'C_OK'))
    class A: pass
    a = A(); a.clf='ridge'
    out = tf.train(a)
    assert called['x'] and out == 'C_OK'
