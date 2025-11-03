import os, sys, tempfile, types


def test_sidecar_written_for_classic(monkeypatch):
    import train_frames as tf
    # Fake tiny dataset
    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, "list_videos", lambda root: vids)
    import vkb.io as vio
    monkeypatch.setattr(vio, "list_videos", lambda root: vids)

    # Fake cv2 and embedder
    class FakeCv2:
        def __init__(self): pass
        class VideoCapture:
            def __init__(self, path):
                import numpy as np
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((8,8,3), val, dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self):
                return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
    monkeypatch.setitem(sys.modules, 'cv2', FakeCv2())

    # Patch the symbol used by train_frames directly
    monkeypatch.setattr(tf, "create_embedder", lambda name: (lambda fr: [0.0, 1.0]))

    # Redirect models dir
    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))

    # Run
    from types import SimpleNamespace
    args = SimpleNamespace(data='.', embed_model='mobilenetv3_small_100', clf='ridge', alpha=1.0, hpo_alpha=0, eval_split=0.2, eval_mode='tail')
    tf.train(args)
    # Check files
    files = os.listdir(tmp)
    assert any(f.endswith(".pkl") for f in files)
    metas = [f for f in files if f.endswith(".pkl.meta.json")]
    assert metas, "expected sidecar .meta.json next to model"
