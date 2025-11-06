import os, tempfile, shutil
import numpy as np


def test_sidecar_contains_mp_params(monkeypatch):
    import train_frames as tf
    # Tiny data
    vids = [("a.mp4", "A"), ("b.mp4", "B")]
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)
    import vkb.io as vio
    monkeypatch.setattr(vio, 'list_videos', lambda root: vids)

    # Monkeypatch feature extractor to avoid mediapipe
    def fake_extract(path, stride=5, max_frames=200):
        base = np.zeros(210, dtype=float)
        if os.path.basename(path).startswith('a'):
            base[0] = 1.0
        else:
            base[1] = 1.0
        return np.stack([base.copy() for _ in range(4)], axis=0)
    import vkb.landmarks as lm
    monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

    # Redirect saves
    tmp = tempfile.mkdtemp(); models = os.path.join(tmp, 'models'); os.makedirs(models)
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, 'save_model', lambda obj, parts, base_dir='models', ext='.pkl': real_save(obj, parts, base_dir=models, ext=ext))
    monkeypatch.setattr(tf, 'save_sidecar', lambda path, meta: __import__('vkb.artifacts').artifacts.save_sidecar(path, meta))
    # Train
    args = tf.parse_args(['--data','.', '--clf','mp_logreg', '--eval-split','0.5', '--C','1.0', '--mp-stride','7', '--mp-max-frames','12'])
    path = tf.train(args)
    # Verify sidecar
    side = path + '.meta.json'
    assert os.path.exists(side)
    import json
    with open(side, 'r') as f:
        m = json.load(f)
    assert m['embed_model'] == 'mediapipe_hand'
    assert m['feat_dim'] == 210
    hp = m['hparams']
    assert hp['C'] > 0
    assert hp['mp_stride'] == 7
    assert hp['mp_max_frames'] == 12
    assert hp['feat_norm'] == 'l1'
    assert hp['pairs'] == 'upper_xy'
    assert hp['landmarks'] == 21
    shutil.rmtree(tmp, ignore_errors=True)

