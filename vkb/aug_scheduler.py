class AugScheduler:
    """Round‑robin dynamic augmentation controller with log‑spaced levels.

    - Each knob (brightness, warp, sat, contrast, hue, wb) has 10 log steps
      between 0 and a generous max; we step up/down by one per change.
    - Tie (val_acc unchanged) keeps the new level; worse reverts to previous.
    """
    def __init__(self, ds_tr, args, methods=None, patience: int = 1, steps: int = 10):
        self.ds = ds_tr
        # Also schedule rotation; cap is full rotation in degrees to match tests.
        self.methods = methods or ["brightness", "warp", "sat", "contrast", "hue", "wb", "shift", "rot_deg"]
        self.maxes = {"brightness": 0.6, "warp": 0.6, "sat": 0.8, "contrast": 0.8, "hue": 0.5, "wb": 0.5, "shift": 0.5, "rot_deg": 360.0}
        self.steps = max(1, int(steps))
        # Build logarithmic levels per knob: [0, l1..lN]
        self.levels = {}
        for k in self.methods:
            mx = float(self.maxes.get(k, 0.5))
            mn = max(mx * 1e-2, 1e-4)
            ratio = (mx / mn) ** (1.0 / self.steps)
            vals = [0.0]
            v = mn
            for _ in range(self.steps):
                vals.append(min(mx, v))
                v *= ratio
            # Snap last level exactly to max to satisfy hard caps like rot_deg=360
            if len(vals) > 1:
                vals[-1] = mx
            self.levels[k] = vals
        # Init all strengths to 0 and track indices
        for k in self.methods:
            setattr(self.ds, k, 0.0)
        self.idx_level = {k: 0 for k in self.methods}
        self.prev = {k: 0.0 for k in self.methods}
        self.idx = 0
        self.patience = int(patience)
        self.noimp = 0
        self.raised = False
        self.monitor = 0
        self.action = None  # 'raise' | 'lower'
        self.p_decrease = 0.25

    def update(self, improved: bool, equal: bool = False):
        if improved:
            self.noimp = 0; self.monitor = 0; self.raised = False
            self.idx = (self.idx + 1) % len(self.methods)
            return
        if not self.raised:
            self.noimp += 1
            if self.noimp >= self.patience:
                import random as _rnd
                k = self.methods[self.idx]
                cur_idx = int(self.idx_level.get(k, 0))
                cur_val = float(getattr(self.ds, k, 0.0) or 0.0)
                self.prev[k] = cur_val
                do_lower = (_rnd.random() < self.p_decrease) and (cur_idx > 0)
                new_idx = max(0, cur_idx - 1) if do_lower else min(self.steps, cur_idx + 1)
                self.action = 'lower' if do_lower else 'raise'
                self.idx_level[k] = new_idx
                setattr(self.ds, k, self.levels[k][new_idx])
                self.raised = True; self.monitor = 0; self.noimp = 0
        else:
            self.monitor += 1
            if self.monitor >= self.patience:
                k = self.methods[self.idx]
                if equal:
                    # tie: keep new value
                    self.raised = False; self.monitor = 0; self.noimp = 0
                    self.idx = (self.idx + 1) % len(self.methods)
                else:
                    # worse: revert to prev and snap index to nearest level
                    prev_val = self.prev.get(k, 0.0)
                    setattr(self.ds, k, prev_val)
                    try:
                        lv = self.levels[k]
                        nearest = min(range(len(lv)), key=lambda i: abs(lv[i]-prev_val))
                        self.idx_level[k] = int(nearest)
                    except Exception:
                        pass
                    self.raised = False; self.monitor = 0; self.noimp = 0
                    self.idx = (self.idx + 1) % len(self.methods)
