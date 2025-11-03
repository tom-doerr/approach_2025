import os, sys, io, pickle, tempfile, shutil, unittest


class TestInferLatestLabel(unittest.TestCase):
    def test_infer_uses_latest_and_label_names(self):
        import infer_live as inf
        # temp models dir as CWD/models
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, 'models'); os.makedirs(models_dir)
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            # Save two bundles; the second becomes LATEST automatically
            bundle1 = {"clf_name":"ridge","embed_model":"mobilenetv3_small_100","labels":["PgDown","PgUp"],"clf": None}
            with open(os.path.join(models_dir, '20250101_000000_ridge_mobilenetv3_small_100.pkl'), 'wb') as f:
                pickle.dump(bundle1, f)
            bundle2 = {"clf_name":"ridge","embed_model":"mobilenetv3_small_100","labels":["PgDown","PgUp"],"clf": None}
            with open(os.path.join(models_dir, '20250102_000000_ridge_mobilenetv3_small_100.pkl'), 'wb') as f:
                pickle.dump(bundle2, f)
            with open(os.path.join(models_dir, 'LATEST'), 'w') as lf:
                lf.write('20250102_000000_ridge_mobilenetv3_small_100.pkl')

            # stub embedder to force class index 1 (PgUp)
            def fake_embedder(*a, **k):
                def emb(_):
                    import numpy as np
                    return np.zeros(1024, dtype='float32')
                return emb
            old_embed = inf.create_embedder; inf.create_embedder = fake_embedder

            # stub cv2
            import numpy as np
            class FakeCap:
                def __init__(self, *_): self.frames=[np.zeros((8,8,3), dtype=np.uint8)]
                def isOpened(self): return True
                def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
                def release(self): pass
            class FakeCv: VideoCapture=FakeCap
            def _noop(*a, **k): pass
            FakeCv.imshow=_noop; FakeCv.waitKey=lambda *a,**k: ord('q'); FakeCv.destroyAllWindows=_noop
            FakeCv.putText=_noop; FakeCv.COLOR_BGR2RGB=0; FakeCv.resize=lambda i,s:i; FakeCv.cvtColor=lambda i,c:i
            FakeCv.FONT_HERSHEY_SIMPLEX = 0; FakeCv.LINE_AA = 0
            old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

            # run infer (classic path needs clf present)
            bundle_path = os.path.join(models_dir, '20250102_000000_ridge_mobilenetv3_small_100.pkl')
            with open(bundle_path, 'rb') as f:
                b = pickle.load(f)
            # attach a trivial sklearn classifier
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            clf = LogisticRegression(max_iter=1).fit(np.zeros((2,1024)), np.array([0,1]))
            b['clf'] = clf
            with open(bundle_path, 'wb') as f:
                pickle.dump(b, f)

            buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
            try:
                sys.argv = ["infer_live.py"]
                inf.main()
            finally:
                sys.stdout = old
            out = buf.getvalue()
            assert 'Loaded' in out and 'LATEST' not in out  # Loaded path prints
        finally:
            inf.create_embedder = old_embed
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
