import io, os, sys, tempfile, shutil, unittest
from types import SimpleNamespace


class TestFinetuneProgressRich(unittest.TestCase):
    def test_rich_progress_does_not_crash(self):
        import train_frames as tf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, 'models'); os.makedirs(models_dir)

        vids = [("vid_A","A"),("vid_B","B")]
        orig_list = tf.list_videos; tf.list_videos = lambda _r: vids
        import vkb.io as vio; orig_lv = vio.list_videos; vio.list_videos = lambda _r: vids

        import numpy as np
        class Cap:
            def __init__(self, p): self.f=[np.zeros((8,8,3),dtype=np.uint8) for _ in range(5)]
            def isOpened(self): return True
            def read(self): return (True, self.f.pop()) if self.f else (False,None)
            def release(self): pass
            def get(self, p): return 5
            def set(self, *a): pass
        import types as _t
        FakeCv=_t.SimpleNamespace(VideoCapture=Cap, resize=lambda i,s:i, cvtColor=lambda i,c:i, COLOR_BGR2RGB=0, CAP_PROP_FRAME_COUNT=7)
        old_cv = sys.modules.get('cv2'); sys.modules['cv2']=FakeCv

        import types
        class TinyNet:
            def __init__(self, num_classes=2):
                import torch, torch.nn as nn
                self.net=nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
            def to(self,d): return self
            def parameters(self): return self.net.parameters()
            def state_dict(self): return self.net.state_dict()
            def __call__(self,x): return self.net(x)
            def train(self): return self
            def eval(self): return self
        old_timm = sys.modules.get('timm'); sys.modules['timm']=types.SimpleNamespace(create_model=lambda n,pretrained=True,num_classes=2:TinyNet(num_classes))

        import vkb.artifacts as va; orig_save=va.save_model
        tf.save_model=lambda obj, parts, base_dir='models', ext='.pkl': orig_save(obj, parts, base_dir=models_dir, ext=ext)

        buf=io.StringIO(); old_out=sys.stdout; sys.stdout=buf
        try:
            args = SimpleNamespace(data='ignored', embed_model='mobilenetv3_small_100', backbone='mobilenetv3_small_100', eval_split=0.2, eval_mode='tail', clf='dl', epochs=1, batch_size=2, lr=1e-3, wd=0.0, device='cpu', rich_progress=True, workers=0)
            tf.train(args)
        finally:
            sys.stdout=old_out
            tf.list_videos=orig_list; vio.list_videos=orig_lv
            if old_cv is not None: sys.modules['cv2']=old_cv
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm']=old_timm
            else: del sys.modules['timm']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__=='__main__':
    unittest.main()
