import os, sys, io, pickle, tempfile, shutil, unittest
import numpy as np


class TestInferLiveMPLogReg(unittest.TestCase):
    def test_infer_mp_logreg_runs_with_stubs(self):
        import infer_live as inf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)
        try:
            # Build a tiny sklearn classifier with correct feature dim (210)
            from sklearn.linear_model import LogisticRegression
            X = np.zeros((4, 210), dtype=float)
            X[:2, 0] = 1.0; X[2:, 1] = 1.0
            y = np.array([0,0,1,1])
            clf = LogisticRegression(max_iter=50).fit(X, y)
            bundle = {"clf_name": "mp_logreg", "labels": ["A","B"], "embed_model": "mediapipe_hand", "clf": clf}
            path = os.path.join(models_dir, "20250101_000000_mp_logreg_mediapipe_hand.pkl")
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
            with open(os.path.join(models_dir, "LATEST"), "w") as f:
                f.write(os.path.basename(path))

            # Stub cv2 and mediapipe
            class FakeCv:
                def VideoCapture(self, *_):
                    class C:
                        def __init__(self): self.cnt = 0
                        def isOpened(self): return True
                        def read(self):
                            if self.cnt >= 3: return False, None
                            self.cnt += 1
                            # 64x64 black frame
                            import numpy as _np
                            return True, _np.zeros((64,64,3), dtype=_np.uint8)
                        def release(self): pass
                    return C()
                def waitKey(self, *_): return ord('q')
                def imshow(self, *_): pass
                def destroyAllWindows(self): pass
                def putText(self, *a, **k): pass
                FONT_HERSHEY_SIMPLEX = 0
                LINE_AA = 0
                def cvtColor(self, img, code): return img
                def resize(self, img, sz): return img
                COLOR_BGR2RGB = 0
                def CAP_PROP_FOURCC(self): return 6
                def VideoWriter_fourcc(self, *a): return 0
                def CAP_PROP_FRAME_WIDTH(self): return 3
                def CAP_PROP_FRAME_HEIGHT(self): return 4
                def CAP_PROP_FPS(self): return 5
                def set(self, *a, **k): pass
            old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

            class _Pt:
                def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z

            class FakeHands:
                def __init__(self, *a, **k): pass
                def process(self, rgb):
                    class R:
                        pass
                    r = R()
                    # 21 landmarks in a line to avoid degenerate zero-sum
                    r.multi_hand_landmarks = [type('LM', (), {'landmark': [_Pt(i/21.0, 0.0, 0.0) for i in range(21)]})]
                    return r
                def close(self): pass

            class FakeMP:
                class solutions:
                    class hands:
                        Hands = FakeHands
            old_mp = sys.modules.get('mediapipe'); sys.modules['mediapipe'] = FakeMP()

            # Run for a few frames, CPU path
            old_argv = sys.argv
            old_env_dir = os.environ.get('VKB_MODELS_DIR')
            os.environ['VKB_MODELS_DIR'] = models_dir
            try:
                sys.argv = ["infer_live.py", "--frames", "3"]
                inf.main()
            finally:
                sys.argv = old_argv
                if old_env_dir is None:
                    os.environ.pop('VKB_MODELS_DIR', None)
                else:
                    os.environ['VKB_MODELS_DIR'] = old_env_dir
        finally:
            # cleanup stubs and temp dir
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            if old_mp is not None: sys.modules['mediapipe'] = old_mp
            else: del sys.modules['mediapipe']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
