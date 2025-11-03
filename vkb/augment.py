"""Minimal augmentation utilities.

Exports:
- _resize_to, _zoom_out, _rotate_square, _perspective_warp_inward
- Augment: small class wrapping the policy used by FrameDataset
"""

def _resize_to(img, w, h):
    import numpy as np
    try:
        import cv2 as cv
        out = cv.resize(img, (w, h))
        if out.shape[0] == h and out.shape[1] == w:
            return out
    except Exception:
        pass
    ih, iw = img.shape[:2]
    ys = (np.linspace(0, ih - 1, h)).astype(int)
    xs = (np.linspace(0, iw - 1, w)).astype(int)
    return img[ys][:, xs]


def _zoom_out(img, out_size: int, scale_min: float, scale_max: float):
    import cv2 as cv, numpy as np, random
    h, w = img.shape[:2]; s = float(random.uniform(scale_min, scale_max))
    target = int(out_size)
    new_w = max(1, int(round(target * s)))
    new_h = max(1, int(round(target * s)))
    resized = _resize_to(img, new_w, new_h)
    if s <= 1.0:
        y0 = (target - new_h) // 2; x0 = (target - new_w) // 2
        top, bottom = y0, target - (y0 + new_h)
        left, right = x0, target - (x0 + new_w)
        return cv.copyMakeBorder(resized, top, bottom, left, right, borderType=cv.BORDER_REFLECT_101)
    y0 = max(0, (new_h - target) // 2); x0 = max(0, (new_w - target) // 2)
    return resized[y0:y0+target, x0:x0+target]


def _rotate_square(img, angle_deg: float):
    import cv2 as cv, numpy as np
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(c, float(angle_deg), 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)


def _perspective_warp_inward(img, max_frac: float):
    import cv2 as cv, numpy as np, random
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    dst = []
    for (x, y) in src:
        f = random.uniform(0.0, float(max_frac))
        dx, dy = (cx - x) * f, (cy - y) * f
        dst.append([x + dx, y + dy])
    try:
        M = cv.getPerspectiveTransform(src, np.float32(dst))
        return cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    except Exception:
        # Minimal fallback: crop a small inward border and resize back.
        from vkb.finetune import _resize_to as _rz
        m = int(max(1, round(min(h, w) * float(max_frac) * 0.1)))
        if m * 2 >= h or m * 2 >= w:
            return img
        cropped = img[m:h - m, m:w - m]
        return _rz(cropped, w, h)


class Augment:
    def __init__(self, size: int, mode: str = 'train', aug: str = 'none', warp: float = 0.0,
                 zoom_out=None, rotate_square=None, perspective_warp_inward=None):
        self.size = int(size)
        self.mode = str(mode)
        self.aug = str(aug)
        self.warp = float(warp)
        self._zoom_out = zoom_out or _zoom_out
        self._rotate_square = rotate_square or _rotate_square
        self._perspective_warp_inward = perspective_warp_inward or _perspective_warp_inward

    def apply(self, img):
        import cv2 as cv
        fr = img
        if self.mode == 'train' and self.aug != 'none':
            if self.aug == 'rot360':
                fr = self._zoom_out(fr, self.size, 0.60, 1.10)
                import random as _random
                fr = self._rotate_square(fr, _random.uniform(0.0, 360.0))
            elif self.aug == 'light':
                fr = self._zoom_out(fr, self.size, 0.90, 1.00)
            else:
                fr = self._zoom_out(fr, self.size, 0.80, 1.00)
        else:
            try:
                fr = cv.resize(fr, (self.size, self.size))
            except Exception:
                fr = _resize_to(fr, self.size, self.size)
        if self.mode == 'train' and self.warp > 0.0:
            fr = self._perspective_warp_inward(fr, self.warp)
        return fr
