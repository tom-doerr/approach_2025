import os, tempfile, shutil
import numpy as np


def test_mp_logreg_default_records_unlimited_maxframes(monkeypatch):
    import train_frames as tf
    import vkb.landmarks as lm
    # dataset
    tmp = tempfile.mkdtemp(); models_tmp = tempfile.mkdtemp()
    try:
        for lab in ('A','B'):
            os.makedirs(os.path.join(tmp, lab), exist_ok=True)
        open(os.path.join(tmp, 'A', 'a.mp4'), 'wb').close()
        open(os.path.join(tmp, 'B', 'b.mp4'), 'wb').close()

        # Spy: capture max_frames passed
        seen = {'max': None}
        def fake_extract(path, stride=5, max_frames=0):
            seen['max'] = max_frames
            v = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'):
                v[0] = 1.0
            else:
                v[1] = 1.0
            return np.stack([v.copy() for _ in range(6)], axis=0)

        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

        old_env = os.environ.get('VKB_MODELS_DIR'); os.environ['VKB_MODELS_DIR'] = models_tmp
        args = tf.parse_args(['--data', tmp, '--clf', 'mp_logreg', '--eval-split', '0.5'])
        path = tf.train(args)
        assert seen['max'] == 0
        import json
        with open(path + '.meta.json', 'r') as f:
            meta = json.load(f)
        assert meta['hparams']['mp_max_frames'] == 0
    finally:
        if old_env is None: os.environ.pop('VKB_MODELS_DIR', None)
        else: os.environ['VKB_MODELS_DIR'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(models_tmp, ignore_errors=True)

