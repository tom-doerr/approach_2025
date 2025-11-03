def test_classic_trainer_exists_and_runs(monkeypatch):
    import train_frames as tf
    assert hasattr(tf, 'ClassicTrainer')

    calls = {'prep': False, 'embed': False, 'fit': False}

    # Stub helpers to avoid heavy work and assert wiring
    monkeypatch.setattr(tf, '_prepare_classic_io', lambda args: (['v1'], ['L'], {'L':0}))
    def _embed(cons, args, vids, labels, lab2i):
        calls['embed'] = True
        return ([1.0], [0], {0:[0]}, 128)
    monkeypatch.setattr(tf, '_embed_videos', _embed)
    monkeypatch.setattr(tf, '_fit_classic_and_save', lambda cons, args, X, Y, labels, idx_by_class, feat_dim: (calls.__setitem__('fit', True) or 'OK_PATH'))

    class A: pass
    a = A(); a.clf = 'ridge'; a.data='.'; a.embed_model='raw'
    out = tf.ClassicTrainer(a).run()
    assert out == 'OK_PATH'
    assert calls['embed'] and calls['fit']
