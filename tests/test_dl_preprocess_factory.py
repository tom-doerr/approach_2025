import unittest
import numpy as np


class TestDLPreprocessFactory(unittest.TestCase):
    def test_preprocess_rgb_resize_normalize(self):
        import infer_live as inf
        # Stub cv2 so we can control resize and color conversion without OpenCV
        class FakeCv:
            COLOR_BGR2RGB = 0
            def resize(self, img, sz):
                # pretend to resize but keep content
                return img
            def cvtColor(self, img, code):
                # swap BGR->RGB
                return img[..., ::-1]
        import sys
        old_cv = sys.modules.get('cv2'); sys.modules['cv2'] = FakeCv()

        try:
            # Build preprocess with identity mean/std to check scaling and channel order
            pp = inf._make_preprocess(2, mean=[0,0,0], std=[1,1,1], dev='cpu')
            # Uniform BGR frame (B=10, G=20, R=30)
            fr = np.full((2,2,3), [10,20,30], dtype=np.uint8)
            x = pp(fr)  # torch tensor [1,3,2,2]
            import torch
            self.assertEqual(tuple(x.shape), (1,3,2,2))
            # Values are scaled to [0,1] and RGB order
            xs = x.squeeze(0)
            self.assertTrue(torch.allclose(xs[0], torch.full((2,2), 30/255.0)))  # R
            self.assertTrue(torch.allclose(xs[1], torch.full((2,2), 20/255.0)))  # G
            self.assertTrue(torch.allclose(xs[2], torch.full((2,2), 10/255.0)))  # B
        finally:
            if old_cv is not None: sys.modules['cv2'] = old_cv
            else: del sys.modules['cv2']


if __name__ == '__main__':
    unittest.main()

