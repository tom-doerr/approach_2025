import io, sys


def test_classic_console_section_order(monkeypatch):
    import numpy as np
    import train_frames as tf

    # Minimal one-video, one-label setup
    vids = [("/tmp/vid.mp4", "L")]
    monkeypatch.setattr(tf, 'list_videos', lambda root: vids)

    # Headless cv2 that yields 2 frames
    class FakeCap:
        def __init__(self, path):
            self.frames = [np.zeros((4,4,3), dtype=np.uint8) for _ in range(2)]
        def isOpened(self): return True
        def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
        def release(self): pass
    class CV: VideoCapture = FakeCap
    old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = CV()

    # Simple embedder (1-D mean)
    monkeypatch.setattr(tf, 'create_embedder', lambda *_a, **_k: (lambda fr: np.array([fr.mean()], dtype=np.float32)))

    # Capture output
    class A: pass
    a = A(); a.data='.'; a.embed_model='raw'; a.clf='ridge'; a.alpha=1.0; a.hpo_alpha=0; a.eval_split=0.0; a.eval_mode='tail'
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        tf.train(a)
    finally:
        sys.stdout = old
        if old_cv is not None: sys.modules['cv2'] = old_cv
        else: del sys.modules['cv2']

    out = buf.getvalue()
    # Lock presence and relative order of key sections
    sections = [
        'Embedding frames from',
        'Class Counts',
        'Cache Summary',
        'Embedding Speed',
        'Training Summary',
    ]
    idx = [-1]
    for s in sections:
        i = out.find(s)
        assert i != -1, f"missing section: {s}"
        assert i > idx[-1], f"section order incorrect around: {s}"
        idx.append(i)

