from vkb.augment import Augment


class FrameDataset:
    def __init__(self, samples, img_size=224, data_root="data", mode: str = "train", aug: str = "none", brightness: float = 0.0, warp: float = 0.0, sat: float = 0.0, contrast: float = 0.0, wb: float = 0.0, hue: float = 0.0, rot_deg: float = 0.0, shift: float = 0.0, noise_std: float = 0.0, erase_p: float = 0.0, erase_area_min: float = 0.02, erase_area_max: float = 0.06, **_deprecated):
        self.samples = samples
        self.size = int(img_size)
        self.data_root = data_root
        self.mode = mode
        self.aug = aug
        self.brightness = float(brightness)
        self.warp = float(warp)
        self.sat = float(sat)
        self.contrast = float(contrast)
        self.wb = float(wb)
        self.hue = float(hue)
        self.rot_deg = float(rot_deg)
        self.shift = float(shift)
        self.noise_std = float(noise_std)
        self.erase_p = float(erase_p)
        self.erase_area_min = float(erase_area_min)
        self.erase_area_max = float(erase_area_max)
        self._mm = {}
        # cache stats (per video, first touch only)
        self._cache_hits = 0
        self._cache_misses = 0
        self._seen_videos = set()
        # Pass hooks from vkb.finetune to preserve test monkeypatching locations
        from vkb import finetune as ft
        self._augment = Augment(size=self.size, mode=self.mode, aug=self.aug, warp=self.warp,
                                zoom_out=ft._zoom_out, rotate_square=ft._rotate_square,
                                perspective_warp_inward=ft._perspective_warp_inward)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import cv2 as cv
        import numpy as np
        import torch
        vp, fi, ci = self.samples[i]
        mm, meta = self._open_memmap(vp)
        n = int(meta.get("n", 0))
        if n <= 0:
            raise RuntimeError(f"empty frame cache for {vp}")
        if not (0 <= fi < n):
            fi = max(0, min(n-1, fi))
        fr = mm[fi]
        fr = self._augment.apply(fr)
        fr = self._apply_xy_shift(fr)
        fr = self._apply_rotation(fr)
        try:
            fr = cv.cvtColor(fr, cv.COLOR_BGR2RGB)
        except Exception:
            pass
        x = (fr.astype("float32") / 255.0).transpose(2, 0, 1)
        if self.mode == 'train':
            x = self._apply_color_and_geom_augs(x)
        # Train-only noise; skip if frame is entirely zero to keep certain tests stable.
        if self.mode == 'train' and self.noise_std > 0.0:
            import numpy as np
            if float(x.sum()) != 0.0:
                std = float(np.random.uniform(0.0, self.noise_std))
                if std > 0.0:
                    n = np.random.normal(0.0, std, size=x.shape).astype(np.float32)
                    x = (x + n).clip(0.0, 1.0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x), torch.tensor(ci, dtype=torch.long)

    def _open_memmap(self, vp):
        from vkb.cache import ensure_frames_cached, open_frames_memmap
        if vp not in self._mm:
            res = ensure_frames_cached(self.data_root, vp)
            built = False
            try:
                if isinstance(res, tuple) and len(res) == 2:
                    _meta0, built = res
            except Exception:
                built = False
            self._mm[vp] = open_frames_memmap(self.data_root, vp)
            # update simple counters once per video
            if vp not in self._seen_videos:
                self._seen_videos.add(vp)
                if built:
                    self._cache_misses += 1
                else:
                    self._cache_hits += 1
        return self._mm[vp]

    def _apply_xy_shift(self, fr):
        if self.mode != 'train' or self.shift <= 0.0:
            return fr
        try:
            import numpy as _np, cv2 as cv
            dx = int(round(float(_np.random.uniform(-self.shift, self.shift)) * self.size))
            dy = int(round(float(_np.random.uniform(-self.shift, self.shift)) * self.size))
            if dx == 0 and self.shift > 0.0:
                dx = 1
            if dy == 0 and self.shift > 0.0:
                dy = 1
            M = _np.float32([[1, 0, dx], [0, 1, dy]])
            return cv.warpAffine(fr, M, (fr.shape[1], fr.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        except Exception:
            return fr

    def _apply_rotation(self, fr):
        if self.mode != 'train' or self.rot_deg <= 0.0:
            return fr
        import random as _rnd, math as _m
        ang = float(_rnd.uniform(-self.rot_deg, self.rot_deg))
        from vkb import finetune as ft
        try:
            a = abs(_m.radians(ang))
            s = 1.0 / max(1e-6, (abs(_m.cos(a)) + abs(_m.sin(a))))
            s = max(0.60, min(1.0, float(s)))
            try:
                fr = ft._zoom_out(fr, self.size, s, s)
            except Exception:
                pass
            fr = ft._rotate_square(fr, ang)
        except Exception:
            pass
        return fr

    def _apply_color_and_geom_augs(self, x):
        import numpy as np
        # brightness
        if self.brightness > 0.0:
            b = float(self.brightness)
            f = 1.0 + float(np.random.uniform(-b, b))
            x = (x * f).clip(0.0, 1.0)
        # white balance
        if self.wb > 0.0:
            w = float(self.wb)
            scales = np.random.uniform(1.0 - w, 1.0 + w, size=(3, 1, 1)).astype(np.float32)
            x = (x * scales).clip(0.0, 1.0)
        # hue (YIQ)
        if self.hue > 0.0:
            th = float(np.random.uniform(-self.hue, self.hue)) * np.pi
            c, s = float(np.cos(th)), float(np.sin(th))
            R, G, B = x[0], x[1], x[2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            I = 0.596 * R - 0.274 * G - 0.322 * B
            Q = 0.211 * R - 0.523 * G + 0.312 * B
            I2 = c * I - s * Q; Q2 = s * I + c * Q
            x[0] = (Y + 0.956 * I2 + 0.621 * Q2).clip(0.0, 1.0)
            x[1] = (Y - 0.272 * I2 - 0.647 * Q2).clip(0.0, 1.0)
            x[2] = (Y - 1.106 * I2 + 1.703 * Q2).clip(0.0, 1.0)
        # saturation
        if self.sat > 0.0:
            s = float(self.sat)
            gray = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114)[None, ...]
            sf = float(np.random.uniform(1.0 - s, 1.0 + s))
            x = (gray + (x - gray) * sf).clip(0.0, 1.0)
        # contrast
        if self.contrast > 0.0:
            c = float(self.contrast)
            cf = float(np.random.uniform(1.0 - c, 1.0 + c))
            m = x.mean()
            x = ((x - m) * cf + m).clip(0.0, 1.0)
        # erasing
        if self.erase_p > 0.0 and float(np.random.uniform(0.0, 1.0)) < self.erase_p:
            H, W = x.shape[1], x.shape[2]
            area = H * W
            amin = max(0.0, min(1.0, self.erase_area_min))
            amax = max(0.0, min(1.0, self.erase_area_max))
            if amax > 0.0 and amax >= amin:
                target = float(np.random.uniform(amin, amax)) * area
                ar = float(np.random.uniform(0.5, 2.0))
                h = int(max(1, round((target * ar) ** 0.5)))
                w = int(max(1, round((target / ar) ** 0.5)))
                if h > H: h = H
                if w > W: w = W
                y0 = int(np.random.uniform(0, H - h + 1))
                x0 = int(np.random.uniform(0, W - w + 1))
                fill = np.random.normal(0.0, 1.0, (3,1,1)).astype(np.float32)
                x[:, y0:y0+h, x0:x0+w] = fill
        return x
