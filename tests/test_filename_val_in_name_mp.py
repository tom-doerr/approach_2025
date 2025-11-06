import os, tempfile, shutil
import numpy as np


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_logreg_filename_no_val_when_eval_zero(monkeypatch):
    import train_frames as tf
    tmp = tempfile.mkdtemp(); models_tmp = tempfile.mkdtemp()
    try:
        _mk_dataset(tmp)
        def fake_extract(path, stride=5, max_frames=200):
            v = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'):
                v[0] = 1.0
            else:
                v[1] = 1.0
            return np.stack([v.copy() for _ in range(3)], axis=0)
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

        old_env = os.environ.get('VKB_MODELS_DIR')
        os.environ['VKB_MODELS_DIR'] = models_tmp

        args = tf.parse_args(['--data', tmp, '--clf', 'mp_logreg', '--eval-split', '0.0', '--C', '1.0'])
        path = tf.train(args)
        bn = os.path.basename(path)
        assert 'val' not in bn
    finally:
        if old_env is None: os.environ.pop('VKB_MODELS_DIR', None)
        else: os.environ['VKB_MODELS_DIR'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(models_tmp, ignore_errors=True)

