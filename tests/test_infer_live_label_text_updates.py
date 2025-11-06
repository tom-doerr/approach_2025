import os, sys, io, pickle, tempfile, shutil, unittest


class FakeCv:
    def __init__(self): self.texts = []
    def VideoCapture(self, *_):
        class C:
            def __init__(self): self.cnt = 0
            def isOpened(self): return True
            def read(self):
                import numpy as _np
                if self.cnt >= 2: return False, None
                self.cnt += 1
                return True, _np.zeros((64,64,3), dtype=_np.uint8)
            def release(self): pass
        return C()
    def waitKey(self, *_): return ord('q')
    def imshow(self, *_): pass
    def destroyAllWindows(self): pass
    def putText(self, img, txt, org, font, scale, color, thick, lineType): self.texts.append(txt)
    def circle(self, *a, **k): pass
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    def cvtColor(self, img, code): return img
    def resize(self, img, sz): return img
    COLOR_BGR2RGB = 0


class TestInferLiveLabelText(unittest.TestCase):
    def test_label_text_not_none(self):
        import infer_live as inf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, 'models'); os.makedirs(models_dir)
        try:
            # Real sklearn LR so predict_proba/decision_function exist
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            clf = LogisticRegression(max_iter=20).fit(np.zeros((3,1)), np.array([0,1,2]))
            labels = ["A","B","C"]
            bundle = {"clf_name":"logreg","embed_model":"mobilenetv3_small_100","labels":labels,"clf": clf}
            path = os.path.join(models_dir, '20250101_000000_logreg_mobilenetv3_small_100.pkl')
            with open(path, 'wb') as f: pickle.dump(bundle, f)
            with open(os.path.join(models_dir, 'LATEST'), 'w') as f: f.write(os.path.basename(path))

            # stub embedder → constant feature; classifier will pick some class
            def fake_embedder(*a, **k):
                def emb(_): return [0.0]
                return emb
            old_embed = inf.create_embedder; inf.create_embedder = fake_embedder
            old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()
            # run
            old_argv = sys.argv
            try:
                sys.argv = ['infer_live.py', '--frames', '1', '--model-path', path]
                inf.main()
            finally:
                sys.argv = old_argv
            texts = sys.modules['cv2'].texts
            # First overlay line contains the predicted label
            label_lines = [t for t in texts if not t.startswith('FPS:') and not t.startswith('Proc:') and not t.startswith('raw=') and not t.startswith('prob=')]
            assert any(t in labels for t in label_lines), f"no label from {labels} in overlay: {label_lines}"
            assert not any(t == 'None' for t in label_lines)
        finally:
            try: inf.create_embedder = old_embed
            except Exception: pass
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

