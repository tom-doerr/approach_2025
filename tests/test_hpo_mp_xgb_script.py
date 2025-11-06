import os, tempfile, shutil
import numpy as np


def _mk_dataset(root: str):
    os.makedirs(os.path.join(root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(root, 'B'), exist_ok=True)
    open(os.path.join(root, 'A', 'a.mp4'), 'wb').close()
    open(os.path.join(root, 'B', 'b.mp4'), 'wb').close()


def test_hpo_mp_xgb_run_with_stubs(monkeypatch):
    import hpo_mp_xgb as hpo
    tmp = tempfile.mkdtemp()
    try:
        _mk_dataset(tmp)
        # Stub features: 210-D one-hot per label, 6 frames each
        def fake_extract(path, stride=1, max_frames=0):
            v = np.zeros(210, dtype=float)
            if os.path.basename(path).startswith('a'): v[0] = 1.0
            else: v[1] = 1.0
            return np.stack([v.copy() for _ in range(6)], axis=0)
        import vkb.landmarks as lm
        monkeypatch.setattr(lm, 'extract_features_for_video', fake_extract)

        res = hpo.run_hpo(tmp, eval_split=0.5, trials=3, seed=0, eval_mode='tail-per-video', mp_stride=1, mp_max_frames=0, use_stub=True)
        assert res['n_samples'] == 12
        assert len(res['trials']) == 3
        # Check presence of latency metrics
        t0 = res['trials'][0]
        for k in ('val_acc','train_ms','infer_ms','infer_sps'): assert k in t0
        # Pareto should have at least one candidate
        assert len(res['pareto_idx']) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
