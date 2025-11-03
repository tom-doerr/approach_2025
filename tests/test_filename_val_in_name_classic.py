import os, sys, tempfile


def test_classic_model_filename_includes_val(monkeypatch):
    import train_frames as tf

    # Tiny two-class, two-frame-per-video dataset
    vids = [("vid_A", "A"), ("vid_B", "B")]
    monkeypatch.setattr(tf, "list_videos", lambda root: vids)
    import vkb.io as vio
    monkeypatch.setattr(vio, "list_videos", lambda root: vids)

    # cv2 + embedder stubs
    class FakeCv2:
        def __init__(self):
            pass
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

    monkeypatch.setattr(tf, "create_embedder", lambda name: (lambda fr: [0.0, 1.0]))

    tmp = tempfile.mkdtemp()
    from vkb.artifacts import save_model as real_save
    monkeypatch.setattr(tf, "save_model", lambda obj, parts, base_dir="models", ext=".pkl": real_save(obj, parts, base_dir=tmp, ext=ext))

    from types import SimpleNamespace
    args = SimpleNamespace(data='.', embed_model='mobilenetv3_small_100', clf='ridge', alpha=1.0, hpo_alpha=0, eval_split=0.5, eval_mode='tail')
    tf.train(args)

    files = [f for f in os.listdir(tmp) if f.endswith('.pkl')]
    assert files, 'expected a saved model file'
    assert any('_val' in f for f in files), 'expected filename to include _val score'
