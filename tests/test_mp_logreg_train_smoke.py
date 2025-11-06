import os, tempfile, shutil
import numpy as np


def _mk_dummy_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_mp_logreg_training_with_monkeypatched_features(monkeypatch):
    # Prepare tiny data tree
    tmp = tempfile.mkdtemp()
    models_tmp = tempfile.mkdtemp()
    try:
        _mk_dummy_dataset(tmp)
        # Monkeypatch feature extractor to avoid mediapipe dependency in tests
        def fake_extract(path, stride=5, max_frames=200):
            # Deterministic, separable features (dim=210)
            base = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'):
                base[0] = 1.0
            else:
                base[1] = 1.0
            # Return 5 copies per video
            return np.stack([base.copy() for _ in range(5)], axis=0)

        monkeypatch.setenv('VKB_MODELS_DIR', models_tmp)
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

        from train_frames import parse_args, train
        args = parse_args([
            '--data', tmp,
            '--clf', 'mp_logreg',
            '--eval-split', '0.5',
            '--C', '1.0',
            '--logreg-max-iter', '100',
        ])
        path = train(args)
        assert os.path.exists(path)
        bn = os.path.basename(path)
        assert 'mp_logreg' in bn
        assert 'val' in bn  # filename contains val score when eval-split>0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(models_tmp, ignore_errors=True)

