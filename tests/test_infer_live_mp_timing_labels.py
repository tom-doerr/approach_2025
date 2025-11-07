import os, sys, io, pickle, tempfile, shutil
import numpy as np


# Top-level picklable stub that looks like XGBClassifier enough for our path
class StubXGB:
    def __init__(self, **p): pass
    def predict_proba(self, X):
        import numpy as _np
        return _np.tile([0.1, 0.9, 0.0], (len(X), 1))


def _mk_bundle_mp_xgb(models_dir):
    bundle = {"clf_name": "mp_xgb", "labels": ["A","B","C"], "embed_model": "mediapipe_hand", "clf": StubXGB()}
    path = os.path.join(models_dir, "20250101_000000_mp_xgb_mediapipe_hand.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(models_dir, "LATEST"), "w") as f:
        f.write(os.path.basename(path))


def _install_cv_stub(captured):
    class FakeCv:
        def VideoCapture(self, *_):
            class C:
                def __init__(self): self.cnt = 0
                def isOpened(self): return True
                def read(self):
                    if self.cnt >= 1: return False, None
                    self.cnt += 1
                    import numpy as _np
                    return True, _np.zeros((64,64,3), dtype=_np.uint8)
                def release(self): pass
                def set(self, *a, **k): pass
            return C()
        def waitKey(self, *_): return ord('q')
        def imshow(self, *_): pass
        def destroyAllWindows(self): pass
        def putText(self, img, text, *a, **k): captured.append(text)
        def circle(self, *a, **k): pass
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0
        def cvtColor(self, img, code): return img
        def resize(self, img, sz): return img
        COLOR_BGR2RGB = 0
        CAP_PROP_FOURCC = 6
        def VideoWriter_fourcc(self, *a): return 0
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
    old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
    return old_cv


def _install_mp_stub():
    class _Pt:
        def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z
    class FakeHands:
        def __init__(self, *a, **k): pass
        def process(self, rgb):
            class R: pass
            r = R()
            r.multi_hand_landmarks = [type('LM', (), {'landmark': [_Pt(i/21.0, 0.0, 0.0) for i in range(21)]})]
            return r
        def close(self): pass
    class FakeMP:
        class solutions:
            class hands:
                Hands = FakeHands
    old_mp = sys.modules.get('mediapipe'); sys.modules['mediapipe'] = FakeMP()
    return old_mp


def test_infer_mp_xgb_timing_labels():
    import infer_live as inf
    tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, 'models'); os.makedirs(models_dir)
    captured = []
    old_cv = None; old_mp = None
    try:
        _mk_bundle_mp_xgb(models_dir)
        old_cv = _install_cv_stub(captured)
        old_mp = _install_mp_stub()
        old_argv = sys.argv; old_env = os.environ.get('VKB_MODELS_DIR')
        os.environ['VKB_MODELS_DIR'] = models_dir
        try:
            sys.argv = ["infer_live.py", "--frames", "1", "--unsafe-load"]
            inf.main()
        finally:
            sys.argv = old_argv
            if old_env is None: os.environ.pop('VKB_MODELS_DIR', None)
            else: os.environ['VKB_MODELS_DIR'] = old_env
    finally:
        if old_cv is not None: sys.modules['cv2'] = old_cv
        elif 'cv2' in sys.modules: del sys.modules['cv2']
        if old_mp is not None: sys.modules['mediapipe'] = old_mp
        elif 'mediapipe' in sys.modules: del sys.modules['mediapipe']
        shutil.rmtree(tmp, ignore_errors=True)
    # Find timing line and ensure MP and XGB labels are present
    timing_lines = [t for t in captured if 'Proc:' in t and 'MP:' in t and 'XGB:' in t]
    assert timing_lines, f"No timing line with MP/XGB labels found: {captured}"
