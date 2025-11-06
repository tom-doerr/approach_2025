import os, tempfile, shutil
import numpy as np


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_logreg_bundle_roundtrip(monkeypatch):
    import train_frames as tf
    import infer_live as inf
    tmp = tempfile.mkdtemp(); models_tmp = tempfile.mkdtemp()
    try:
        _mk_dataset(tmp)
        # Stub features: class A -> e0, class B -> e1
        def fake_extract(path, stride=5, max_frames=200):
            v = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'):
                v[0] = 1.0
            else:
                v[1] = 1.0
            return np.stack([v.copy() for _ in range(6)], axis=0)
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

        old_env = os.environ.get('VKB_MODELS_DIR')
        os.environ['VKB_MODELS_DIR'] = models_tmp

        args = tf.parse_args(['--data', tmp, '--clf', 'mp_logreg', '--eval-split', '0.5', '--C', '1.0'])
        path = tf.train(args)
        assert os.path.exists(path)

        # Load via infer_live.load_bundle and predict a known feature
        b = inf.load_bundle(path)
        assert b['clf_name'] == 'mp_logreg'
        assert b['embed_model'] == 'mediapipe_hand'
        clf = b['clf']; labels = b['labels']

        xB = np.zeros(210, dtype=float); xB[1] = 1.0
        pred = clf.predict([xB])[0]
        assert labels[int(pred)] in ('B', 'b')
    finally:
        if old_env is None: os.environ.pop('VKB_MODELS_DIR', None)
        else: os.environ['VKB_MODELS_DIR'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(models_tmp, ignore_errors=True)

