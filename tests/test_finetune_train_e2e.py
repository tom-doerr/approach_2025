import os, sys, glob, pickle, shutil, tempfile, unittest
from types import SimpleNamespace


class TinyNet:
    # very small conv net compatible with timm.create_model signature
    def __init__(self, num_classes=2):
        import torch
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def __call__(self, x):
        return self.net(x)

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self


class TestFinetuneE2E(unittest.TestCase):
    def test_finetune_end_to_end_cpu(self):
        # Patch list_videos to supply fake labeled videos
        import train_frames as tf
        tmp = tempfile.mkdtemp()
        models_dir = os.path.join(tmp, "models"); os.makedirs(models_dir)

        vids = [("vid_A", "A"), ("vid_B", "B")]
        # patch both the CLI module and the io helper used by finetune
        orig_list_videos_tf = tf.list_videos
        tf.list_videos = lambda _root: vids
        import vkb.io as vio
        orig_list_videos_io = vio.list_videos
        vio.list_videos = lambda _root: vids

        # Fake cv2 VideoCapture reading fixed-size frames
        import numpy as np
        class FakeCap:
            def __init__(self, path):
                # 6 frames per video; A dark, B bright
                val = 32 if path.endswith("A") else 224
                self.frames = [np.full((32, 32, 3), val, dtype=np.uint8) for _ in range(6)]
                self._pos = 0
            def isOpened(self): return True
            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop()
            def release(self): pass
            def get(self, prop):
                # CAP_PROP_FRAME_COUNT
                return 6
            def set(self, prop, value):
                pass

        import types as _types
        def _resize(img, sz):
            return img
        def _cvt(img, code):
            return img
        FakeCv2 = _types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=FakeCap,
            resize=_resize,
            cvtColor=_cvt,
            COLOR_BGR2RGB=0,
        )

        old_cv2 = sys.modules.get('cv2')
        sys.modules['cv2'] = FakeCv2

        # Stub timm.create_model to avoid downloads
        import types
        fake_timm = types.SimpleNamespace(create_model=lambda name, pretrained=True, num_classes=2: TinyNet(num_classes))
        old_timm = sys.modules.get('timm')
        sys.modules['timm'] = fake_timm

        # Redirect model saving to tmp models dir
        orig_save_model = tf.save_model
        import vkb.artifacts as vart
        orig_save_model_vart = vart.save_model
        tf.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_save_model_vart(obj, name_parts, base_dir=models_dir, ext=ext)
        vart.save_model = lambda obj, name_parts, base_dir="models", ext=".pkl": orig_save_model_vart(obj, name_parts, base_dir=models_dir, ext=ext)

        try:
            args = SimpleNamespace(
                data="ignored", embed_model="mobilenetv3_small_100", backbone="mobilenetv3_small_100",
                eval_split=0.2, eval_mode='tail',
                clf="dl", epochs=1, batch_size=4, lr=1e-3, wd=0.0, device='cpu'
            )
            tf.train(args)
            files = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
            self.assertTrue(files, "no model saved")
            with open(files[-1], "rb") as f:
                bundle = pickle.load(f)
            self.assertEqual(bundle["clf_name"], "finetune")
            self.assertEqual(set(bundle["labels"]), {"A","B"})
            self.assertIn("state_dict", bundle)
        finally:
            tf.list_videos = orig_list_videos_tf
            import vkb.io as vio2
            vio2.list_videos = orig_list_videos_io
            tf.save_model = orig_save_model
            import vkb.artifacts as vart2
            vart2.save_model = orig_save_model_vart
            if old_cv2 is not None: sys.modules['cv2'] = old_cv2
            else: del sys.modules['cv2']
            if old_timm is not None: sys.modules['timm'] = old_timm
            else: del sys.modules['timm']
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
