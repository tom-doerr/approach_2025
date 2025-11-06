import os, tempfile, shutil
import numpy as np


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_xgb_training_with_stub_features(monkeypatch):
    import train_frames as tf
    tmp = tempfile.mkdtemp(); models_tmp = tempfile.mkdtemp()
    try:
        _mk_dataset(tmp)
        def fake_extract(path, stride=1, max_frames=0):
            v = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'):
                v[0] = 1.0
            else:
                v[1] = 1.0
            return np.stack([v.copy() for _ in range(10)], axis=0)
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)
        old_env = os.environ.get('VKB_MODELS_DIR'); os.environ['VKB_MODELS_DIR'] = models_tmp
        args = tf.parse_args(['--data', tmp, '--clf', 'mp_xgb', '--eval-split', '0.5', '--hpo-xgb', '0'])
        path = tf.train(args)
        assert os.path.exists(path)
        assert 'mp_xgb' in os.path.basename(path)
        assert 'val' in os.path.basename(path)
    finally:
        if old_env is None: os.environ.pop('VKB_MODELS_DIR', None)
        else: os.environ['VKB_MODELS_DIR'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(models_tmp, ignore_errors=True)

