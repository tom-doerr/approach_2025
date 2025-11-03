import unittest, tempfile, shutil, os, sys
from types import SimpleNamespace


class TestRequireCudaFlag(unittest.TestCase):
    def test_require_cuda_raises_on_cpu(self):
        import train_frames as tf
        tmp = tempfile.mkdtemp(); models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        vids = [("vid_A", "A"), ("vid_B", "B")]
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_io = vio.list_videos
        vio.list_videos = lambda _root: vids

        import numpy as np
        class FakeCap:
            def __init__(self, path):
                self.frames = [np.zeros((8,8,3), dtype=np.uint8) for _ in range(2)]
            def isOpened(self): return True
            def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
            def release(self): pass
            def get(self, prop): return 2
            def set(self, prop, value): pass

        import types as _types
        FakeCv2 = _types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=FakeCap,
            resize=lambda img, sz: img,
            cvtColor=lambda img, code: img,
            COLOR_BGR2RGB=0,
        )
        old_cv2 = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv2

        import types
        class TinyNet:
            def __init__(self, num_classes=2):
                import torch, torch.nn as nn
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(8*8*3, num_classes))
            def to(self, d): return self
            def parameters(self): return self.net.parameters()
            def state_dict(self): return self.net.state_dict()
            def __call__(self, x): return self.net(x)
            def train(self): return self
            def eval(self): return self
        fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=True, num_classes=2: TinyNet(num_classes))
        old_timm = sys.modules.get('timm'); sys.modules['timm'] = fake_timm

        # redirect save
        orig_save_model = tf.save_model
        import vkb.artifacts as vart
        orig_save_model_vart = vart.save_model
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_save_model_vart(obj, name_parts, base_dir=models_dir, ext=ext)
        vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_save_model_vart(obj, name_parts, base_dir=models_dir, ext=ext)

        try:
            args = SimpleNamespace(data="ignored", embed_model="mobilenetv3_small_100", backbone="mobilenetv3_small_100",
                                   eval_split=0.2, eval_mode='tail',
                                   clf="dl", epochs=1, batch_size=2, lr=1e-3, wd=0.0, device='cpu', require_cuda=True)
            with self.assertRaises(RuntimeError):
                tf.train(args)
        finally:
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2; vio2.list_videos = orig_list_videos_io
            tf.save_model = orig_save_model
            import vkb.artifacts as vart2; vart2.save_model = orig_save_model_vart
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            shutil.rmtree(models_dir, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
