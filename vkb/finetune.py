import os
from dataclasses import dataclass
from vkb.augment import (
    _resize_to,
    _zoom_out,
    _rotate_square,
    _perspective_warp_inward,
    Augment,
)
from vkb.vis import setup_visdom as _setup_visdom, visdom_prepare as _visdom_prepare, viz_scalar as _viz_scalar, viz_aug_text as _viz_aug_text
from vkb.dataset import FrameDataset
from vkb.telemetry import Telemetry

# Treat validation accuracies within this epsilon as a "tie".
VAL_EQ_EPS = 2e-3  # 0.002 absolute acc


@dataclass
class FTArgs:
    data: str
    embed_model: str
    eval_split: float
    eval_mode: str
    epochs: int
    batch_size: int
    lr: float
    wd: float
    device: str
    backbone: str = "mobilenetv3_small_100"


def _device_from_arg(arg: str):
    import torch
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _val_compare(val_acc: float | None, best_val: float | None, eps: float = VAL_EQ_EPS):
    """Return (improved, equal) using an epsilon tie.

    - improved: new strictly better than best by > eps
    - equal: |delta| <= eps → treat as tie (keep aug change)
    """
    if val_acc is None:
        return False, False
    if best_val is None:
        return True, False
    d = float(val_acc) - float(best_val)
    if d > float(eps):
        return True, False
    if abs(d) <= float(eps):
        return False, True
    return False, False


def _labels_and_videos(data_root: str):
    from vkb.io import list_videos
    vids = list_videos(data_root)
    labels = sorted({l for _, l in vids})
    lab2i = {l: i for i, l in enumerate(labels)}
    return vids, labels, lab2i


def _build_samples(vids, lab2i, stride: int = 1, stride_offset: int = 0):
    # samples: list of (video_path, frame_idx, class_idx)
    import cv2 as cv
    samples = []
    for vp, lab in vids:
        # Fallback: if CAP_PROP_FRAME_COUNT is unavailable under stubs, estimate by reading
        n = 0
        cap = cv.VideoCapture(vp)
        if not cap.isOpened():
            continue
        try:
            n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        except Exception:
            n = 0
        if not n:
            c = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                c += 1
            n = c
            cap.release(); cap = None
        if cap is not None:
            cap.release()
        if n <= 0:
            continue
        ci = lab2i[lab]
        step = max(1, int(stride))
        off = int(stride_offset) % step
        start = off if off < n else 0
        for fi in range(start, n, step):
            samples.append((vp, fi, ci))
    return samples


def _tail_split(samples, n_classes: int, eval_frac: float):
    # chronological per-class tail split using insertion order
    idx_by_class = {i: [] for i in range(n_classes)}
    for idx, (_vp, _fi, ci) in enumerate(samples):
        idx_by_class[ci].append(idx)
    tr, va = [], []
    for ci in range(n_classes):
        arr = idx_by_class[ci]
        c = len(arr)
        if c == 0:
            continue
        if eval_frac <= 0:
            tr += arr
            continue
        if c < 2:
            v = 0
        else:
            v = max(1, min(c - 1, int(round(c * eval_frac))))
        if v == 0:
            tr += arr
        else:
            tr += arr[:-v]; va += arr[-v:]
    return tr, va

def _tail_split_three(samples, n_classes: int, val_frac: float, test_frac: float):
    # Per-class chronological split: head=train, mid=val, tail=test
    idx_by_class = {i: [] for i in range(n_classes)}
    for idx, (_vp, _fi, ci) in enumerate(samples):
        idx_by_class[ci].append(idx)
    tr, va, te = [], [], []
    for ci in range(n_classes):
        arr = idx_by_class[ci]
        c = len(arr)
        if c == 0:
            continue
        tv = max(0, int(round(c * float(test_frac or 0.0))))
        vv = max(0, int(round(c * float(val_frac or 0.0))))
        total_tail = min(c, tv + vv)
        head = max(0, c - total_tail)
        tr += arr[:head]
        if vv > 0:
            va += arr[head:head+vv]
        if tv > 0:
            te += arr[head+vv:head+vv+tv]
    return tr, va, te

def _per_video_tail_split_three(samples, val_frac: float, test_frac: float):
    """Per‑video chronological split with global fractions (Hamilton rounding).

    - Allocates total val/test frames across videos proportionally to video length,
      then within each video uses head=train, mid=val, tail=test.
    - Reserves one train frame per video when possible by capping val.
    """
    from collections import defaultdict
    import math
    by_vid = defaultdict(list)
    for idx, (vp, _fi, _ci) in enumerate(samples):
        by_vid[vp].append(idx)
    vids = list(by_vid.keys())
    arrs = [by_vid[v] for v in vids]
    lens = [len(a) for a in arrs]
    if not lens or sum(lens) == 0:
        return [], [], []

    def _apportion(counts, frac, cap=None):
        frac = float(frac or 0.0)
        targets = [counts[i] * frac for i in range(len(counts))]
        base = [int(math.floor(t)) for t in targets]
        if cap is not None:
            base = [min(base[i], int(cap[i])) for i in range(len(base))]
        need = int(round(sum(counts) * frac)) - sum(base)
        if need <= 0:
            return base
        rem = [(i, targets[i] - math.floor(targets[i])) for i in range(len(counts))]
        rem.sort(key=lambda x: x[1], reverse=True)
        for i, _r in rem:
            if need <= 0:
                break
            if cap is not None and base[i] >= int(cap[i]):
                continue
            base[i] += 1; need -= 1
        return base

    test_counts = _apportion(lens, test_frac)
    cap_for_val = [max(0, lens[i] - test_counts[i] - 1) for i in range(len(lens))]
    val_counts = _apportion(lens, val_frac, cap=cap_for_val)
    tr, va, te = [], [], []
    for arr, c, vv, tv in zip(arrs, lens, val_counts, test_counts):
        if c <= 0:
            continue
        head = max(0, c - (vv + tv))
        if head > 0:
            tr += arr[:head]
        if vv > 0:
            va += arr[head:head+vv]
        if tv > 0:
            te += arr[-tv:]
    return tr, va, te


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
    # Generalized zoom: if s < 1 → shrink + mirror‑pad; if s > 1 → enlarge + center‑crop.
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
    # s > 1: center‑crop back to target size
    y0 = max(0, (new_h - target) // 2); x0 = max(0, (new_w - target) // 2)
    return resized[y0:y0+target, x0:x0+target]


def _rotate_square(img, angle_deg: float):
    # rotate around center, keep same square size
    import cv2 as cv, numpy as np
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(c, float(angle_deg), 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)


def _perspective_warp_inward(img, max_frac: float):
    # Move each corner toward the center by a random fraction in [0, max_frac]
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
        # Minimal fallback for environments/tests where cv lacks perspective ops:
        # crop a small inward border and resize back; changes pixels without heavy deps.
        m = int(max(1, round(min(h, w) * float(max_frac) * 0.1)))
        if m * 2 >= h or m * 2 >= w:
            return img
        cropped = img[m:h - m, m:w - m]
        return _resize_to(cropped, w, h)


## FrameDataset moved to vkb/dataset.py; imported above for compatibility


## Augment class moved to vkb/augment.py; helpers re-exported via imports above


@dataclass
class Finetuner:
    """Tiny OO API for finetuning.

    For now, this class intentionally delegates to the module-level
    `finetune(args)` to avoid logic duplication. The classmethod
    `finetune()` is provided so callers can use a class-oriented entrypoint.
    """
    args: object

    def fit(self):
        return finetune(self.args)

    def run(self):  # backward‑compat
        return self.fit()

    @classmethod
    def finetune(cls, args):
        return cls(args).fit()


class SampleIndex:
    """Minimal index for videos → samples → tail split.

    Keeps code tiny by reusing existing module helpers.
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.vids = []
        self.labels = []
        self.lab2i = {}
        self.samples = []

    def load(self, stride: int = 1, stride_offset: int = 0):
        vids, labels, lab2i = _labels_and_videos(self.data_root)
        self.vids, self.labels, self.lab2i = vids, labels, lab2i
        self.samples = _build_samples(vids, lab2i, stride=stride, stride_offset=stride_offset)
        return self

    def tail_split(self, eval_frac: float):
        return _tail_split(self.samples, len(self.labels), eval_frac)

    def counts(self):
        # frames per class from samples
        from collections import Counter
        c = Counter(ci for _vp, _fi, ci in self.samples)
        return [c.get(i, 0) for i in range(len(self.labels))]


def _create_timm_model(name: str, num_classes: int, drop_rate: float, drop_path_rate: float, pretrained: bool = True):
    import timm, inspect
    kw = {"pretrained": pretrained, "num_classes": num_classes}
    try:
        sig = inspect.signature(timm.create_model)
        if "drop_rate" in sig.parameters:
            kw["drop_rate"] = float(drop_rate)
        if "drop_path_rate" in sig.parameters:
            kw["drop_path_rate"] = float(drop_path_rate)
    except Exception:
        # minimalist: if signature inspection fails, pass only basic args
        pass
    return timm.create_model(name, **kw)

def _compute_class_weights(samples, tr_idx, n_classes: int):
    # Balanced weights: N / (C * n_c)
    counts = [0]*n_classes
    for i in tr_idx:
        _, _, ci = samples[i]
        counts[ci] += 1
    total = sum(counts) if counts else 0
    if total == 0:
        return None
    weights = []
    for c in range(n_classes):
        nc = counts[c]
        w = (total / (n_classes * nc)) if nc > 0 else 0.0
        weights.append(w)
    return weights

def _log_aug_samples_if_first_batch(bi, xb, args, cons, viz, viz_wins, ep, mlflow=None, telemetry: Telemetry | None = None):
    if bi != 1:
        return
    vcount = int(getattr(args, 'visdom_aug', 0) or 0)
    if vcount <= 0:
        return
    try:
        nlog = int(min(vcount, xb.size(0)))
        xcpu = xb[:nlog].detach().to('cpu')
        import torch as _torch
        mean = _torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = _torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        imgs = (xcpu*std + mean).clamp(0,1)
        if telemetry is not None:
            telemetry.images(imgs)
        else:
            win_id = 'vkb_aug'
            if viz_wins is not None and 'aug_images' in viz_wins and viz_wins['aug_images']:
                viz.images(imgs.numpy(), nrow=min(nlog, 8), win=win_id)
            else:
                viz.images(imgs.numpy(), nrow=min(nlog, 8), win=win_id, opts={"title": "Aug Samples"})
                if viz_wins is not None:
                    viz_wins['aug_images'] = win_id
            if mlflow is not None:
                try:
                    import os, tempfile, numpy as _np, cv2 as _cv
                    N, C, H, W = imgs.shape
                    ncol = min(nlog, 8); rows = (N + ncol - 1)//ncol
                    grid = _np.zeros((rows*H, ncol*W, 3), dtype=_np.uint8)
                    arr = (imgs.cpu().numpy()*255.0).clip(0,255).astype('uint8')
                    for i in range(N):
                        r, c = i//ncol, i % ncol
                        im = arr[i].transpose(1,2,0)
                        grid[r*H:(r+1)*H, c*W:(c+1)*W] = im[:, :, ::-1]
                    tmpdir = tempfile.mkdtemp(prefix='mlflow_aug_')
                    outp = os.path.join(tmpdir, f'aug_ep{ep}.png')
                    _cv.imwrite(outp, grid)
                    mlflow.log_artifact(outp)
                except Exception:
                    pass
    except Exception as _e:
        cons.print(f"[dim]visdom log failed: {type(_e).__name__}: {_e}[/]")

def _forward_backward_step(model, opt, loss_fn, xb, yb, use_events):
    import time as _time
    import torch as _torch
    opt.zero_grad(set_to_none=True)
    if use_events:
        start_evt = _torch.cuda.Event(enable_timing=True)
        end_evt = _torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        gpu_t0 = _time.perf_counter()
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward(); opt.step()
    if use_events:
        end_evt.record(); end_evt.synchronize(); gpu_ms = float(start_evt.elapsed_time(end_evt))
    else:
        gpu_ms = (_time.perf_counter() - gpu_t0) * 1000.0
    return logits, float(loss.item()), float(gpu_ms)

def _print_batch_progress(cons, bi, total_batches, loss_sum, correct, total, seen, t0, load_ms, gpu_ms):
    import time as _time
    dt = max(_time.perf_counter() - t0, 1e-6)
    spd = seen / dt
    cons.print(f"[dim]  {bi}/{total_batches} loss={loss_sum/bi:.3f} acc={correct/max(1,total):.3f} ({spd:.1f} samples/s) io={load_ms:.1f}ms gpu={gpu_ms:.1f}ms[/]")


def _setup_mlflow(args):
    import os as _os
    mlflow = None
    if bool(getattr(args, 'mlflow', False)) and _os.getenv('VKB_MLFLOW_DISABLE','') != '1':
        try:
            import mlflow as _mlf
        except Exception as e:
            raise RuntimeError("--mlflow set but mlflow is not installed") from e
        if getattr(args, 'mlflow_uri', None):
            _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(getattr(args, 'mlflow_exp', 'vkb'))
        run_name = getattr(args, 'mlflow_run_name', None) or f"dl|{args.backbone}"
        _mlf.start_run(run_name=run_name)
        mlflow = _mlf
    return mlflow


def _run_training_epochs(args, dl_train, dl_val, model, device, opt, loss_fn, cons, prog, viz, viz_wins, labels, n_classes, mlflow):
    best_val = None; ep_times = []; total_seen = 0
    def _one(ep):
        nonlocal best_val, ep_times, total_seen
        tr_acc, seen, io_t, gpu_t, bat_t, ep_dt = _train_epoch(ep, args, dl_train, model, device, opt, loss_fn, cons, prog, viz, viz_wins, mlflow=mlflow)
        msg = f"epoch {ep}/{args.epochs} train_acc={tr_acc:.3f}"
        val_acc, _ = _validate_epoch(dl_val, model, device, n_classes, labels, cons, loss_fn=loss_fn)
        if val_acc is not None:
            msg += f" val_acc={val_acc:.3f}"; best_val = val_acc
        cons.print(msg)
        if mlflow is not None:
            try:
                mlflow.log_metric('train_acc', float(tr_acc), step=ep)
                if val_acc is not None:
                    mlflow.log_metric('val_acc', float(val_acc), step=ep)
            except Exception:
                pass
        _viz_scalar(viz, viz_wins, "train", ep, tr_acc, "Train/Val Acc")
        if val_acc is not None:
            _viz_scalar(viz, viz_wins, "val", ep, val_acc, "Train/Val Acc")
        _viz_aug_text(viz, viz_wins, args, ep)
        ep_times.append(ep_dt); total_seen += seen
        _print_perf(io_t, gpu_t, bat_t, cons, int(getattr(args, 'batch_size', 0) or 0))
    if prog:
        with prog:
            outer = prog.add_task("Epochs", total=args.epochs)
            for ep in range(1, args.epochs + 1):
                _one(ep)
                prog.advance(outer, 1)
    else:
        for ep in range(1, args.epochs + 1):
            _one(ep)
    return best_val, ep_times, total_seen


def _print_summary(ep_times, total_seen, cons):
    if not ep_times:
        return
    total_time = sum(ep_times)
    avg_epoch = total_time / len(ep_times)
    train_fps = total_seen / total_time if total_time > 0 else 0.0
    cons.print(f"[dim]Summary: epochs={len(ep_times)} avg_epoch={avg_epoch:.2f}s total={total_time:.2f}s train_fps={train_fps:.1f} samples/s[/]")


def _mlflow_log_params(args, mlflow):
    if mlflow is None:
        return
    try:
        mlflow.log_params({
            'backbone': args.backbone,
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.wd),
            'eval_split': float(args.eval_split),
            'eval_mode': str(args.eval_mode),
            'drop_path': float(getattr(args,'drop_path',0.0) or 0.0),
            'dropout': float(getattr(args,'dropout',0.0) or 0.0),
            'aug': str(getattr(args,'aug','none')),
            'brightness': float(getattr(args,'brightness',0.0) or 0.0),
            'warp': float(getattr(args,'warp',0.0) or 0.0),
        })
    except Exception:
        pass


def _assert_device_flags(args, device):
    import torch
    if getattr(args, 'require_cuda', False):
        if not (str(device).startswith('cuda') and torch.cuda.is_available()):
            raise RuntimeError("CUDA required but not available. Install GPU PyTorch and use --device cuda.")


def _init_progress(args):
    if bool(getattr(args, 'rich_progress', False)):
        from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        return Progress("[progress.description]{task.description}", BarColumn(), "{task.completed}/{task.total}", "{task.percentage:>3.0f}%", TimeElapsedColumn(), TimeRemainingColumn())
    return None


from vkb.aug_scheduler import AugScheduler


def _save_epoch_model(cons, args, model, labels, ep: int):
    # Minimal per-epoch checkpoint: same bundle schema as final save.
    try:
        from vkb.artifacts import save_model as _save_model
        bundle = {
            "clf_name": "finetune",
            "labels": labels,
            "model_name": args.backbone,
            "state_dict": model.state_dict(),
            "input_size": int(getattr(args, 'img_size', 224) or 224),
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        path = _save_model(bundle, ["finetune", args.backbone, f"ep{int(ep):02d}"] , base_dir="models")
        try:
            # Print path on its own line for easy copy/paste
            cons.print(path)
        except Exception:
            pass
    except Exception:
        pass


def _setup_viz_and_telemetry(args, cons, mlflow):
    viz = _setup_visdom(args, cons)
    viz_wins = {}
    if viz is not None:
        _visdom_prepare(viz)
    from vkb.telemetry import Telemetry
    telemetry = Telemetry(cons, viz=viz, mlflow=mlflow)
    return viz, viz_wins, telemetry


def _epoch_after(cons, telemetry, ep, args, tr_acc, val_acc):
    telemetry.metric('train_acc', float(tr_acc), step=ep)
    try:
        mlf = getattr(telemetry, 'mlflow', None)
        if mlf is not None:
            mlf.log_metric('train_acc', float(tr_acc), step=ep)
    except Exception:
        pass
    if val_acc is not None:
        telemetry.metric('val_acc', float(val_acc), step=ep)
        try:
            mlf = getattr(telemetry, 'mlflow', None)
            if mlf is not None:
                mlf.log_metric('val_acc', float(val_acc), step=ep)
        except Exception:
            pass
    telemetry.scalar("train", ep, tr_acc, "Train/Val Acc")
    if val_acc is not None:
        telemetry.scalar("val", ep, val_acc, "Train/Val Acc")
    telemetry.text(
        f"epoch {ep}/{getattr(args,'epochs',ep)}: "
        f"aug={getattr(args,'aug','none')} "
        f"brightness={getattr(args,'brightness',0.0)} "
        f"warp={getattr(args,'warp',0.0)} "
        f"drop_path={getattr(args,'drop_path',0.0)} "
        f"dropout={getattr(args,'dropout',0.0)}"
    )
    # Dynamic-aug: plot current aug strengths as lines when Visdom is active
    try:
        if bool(getattr(args, 'dynamic_aug', False)) and getattr(args, '_ds_tr', None) is not None:
            ds = args._ds_tr
            key_map = {
                "brightness": "brightness",
                "warp": "warp",
                "shift": "shift",
                "sat": "saturation",
                "contrast": "contrast",
                "hue": "hue",
                "wb": "white_balance",
                "rot_deg": "rotation_deg",
            }
            for k in ("brightness","warp","shift","sat","contrast","hue","wb","rot_deg"):
                v = float(getattr(ds, k, 0.0) or 0.0)
                name = key_map.get(k, k)
                telemetry.scalar2("aug_strengths", name, ep, v, title="Policy (Aug + Reg)", ylabel="value", win="vkb_policy_strengths")
            # Also plot regularization knobs if present
            try:
                doval = float(getattr(args, 'dropout', 0.0) or 0.0)
                telemetry.scalar2("aug_strengths", "dropout", ep, doval, title="Policy (Aug + Reg)", ylabel="value", win="vkb_policy_strengths")
            except Exception:
                pass
            try:
                dpval = float(getattr(args, 'drop_path', 0.0) or 0.0)
                telemetry.scalar2("aug_strengths", "drop_path", ep, dpval, title="Policy (Aug + Reg)", ylabel="value", win="vkb_policy_strengths")
            except Exception:
                pass
        # After first epoch: print a tiny cache summary (videos/hits/misses).
        # With num_workers>0, per-dataset counters live in worker processes, so
        # compute a reliable snapshot by probing caches for the known videos.
        if int(ep) == 1 and getattr(args, '_vids', None):
            try:
                from vkb.cache import ensure_frames_cached as _ens
                vcnt = 0; hits = 0; misses = 0
                for vp, _lab in (getattr(args, '_vids', []) or []):
                    vcnt += 1
                    try:
                        _meta, built = _ens(getattr(args, 'data', 'data'), vp)
                        if built:
                            misses += 1
                        else:
                            hits += 1
                    except Exception:
                        misses += 1
                cons.print(f"[dim]Cache: videos={vcnt} hits={hits} misses={misses}[/]")
            except Exception:
                pass
        # Optional lightweight system diagnostics (opt-in to avoid noisy tests)
        if bool(getattr(args, 'diag_system', False)):
            try:
                cons.print(_system_diag_line())
            except Exception:
                pass
    except Exception:
        pass

def _system_diag_line():
    """Return a short [dim]Sys: ...[/] line with loadavg and thread counts.
    Linux-focused; degrades quietly elsewhere.
    """
    import os
    load = "n/a"
    try:
        with open("/proc/loadavg","r") as f:
            load = f.read().strip().split()[:3]
            load = "/".join(load)
    except Exception:
        pass
    env = os.environ
    omp = env.get("OMP_NUM_THREADS","-")
    mkl = env.get("MKL_NUM_THREADS","-")
    obl = env.get("OPENBLAS_NUM_THREADS","-")
    nex = env.get("NUMEXPR_NUM_THREADS","-")
    cvt = "-"
    try:
        import cv2 as _cv
        try:
            cvt = str(_cv.getNumThreads())
        except Exception:
            pass
    except Exception:
        pass
    tth = "-"
    try:
        import torch as _torch
        try:
            tth = str(_torch.get_num_threads())
        except Exception:
            pass
    except Exception:
        pass
    return f"[dim]Sys: loadavg={load} torch_thr={tth} cv_thr={cvt} OMP={omp} MKL={mkl} OPENBLAS={obl} NUMEXPR={nex}[/]"

def _train_epoch(ep, args, dl_train, model, device, opt, loss_fn, cons, prog, viz=None, viz_wins=None, mlflow=None, telemetry: Telemetry | None = None):
    import time as _time
    import torch as _torch
    cons.print(f"[dim]Epoch {ep}/{args.epochs} (batches: {len(dl_train)})[/]")
    if prog:
        epoch_task = prog.add_task(f"Epoch {ep}", total=len(dl_train))
    t0 = _time.perf_counter(); seen = 0; correct = 0; total = 0; loss_sum = 0.0
    step_print = max(1, len(dl_train)//5)
    prev_end = t0
    use_events = str(device).startswith('cuda') and _torch.cuda.is_available()
    io_times, gpu_times, batch_times = [], [], []
    model.train()
    for bi, (xb, yb) in enumerate(dl_train, 1):
        load_ms = (_time.perf_counter() - prev_end) * 1000.0
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        _log_aug_samples_if_first_batch(bi, xb, args, cons, viz, viz_wins, ep, mlflow=mlflow, telemetry=telemetry)
        logits, loss_val, gpu_ms = _forward_backward_step(model, opt, loss_fn, xb, yb, use_events)
        total += yb.size(0); seen += yb.size(0); loss_sum += float(loss_val)
        io_times.append(load_ms); gpu_times.append(gpu_ms); batch_times.append(load_ms + gpu_ms)
        with _torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
        # Optional live validation on a shuffled subset every N batches
        try:
            vli = int(getattr(args, 'val_live_interval', 0) or 0)
        except Exception:
            vli = 0
        if vli > 0 and (bi % vli == 0):
            _live_val_step(args, model, device, telemetry, cons, ep=ep, global_step=((ep-1)*len(dl_train)+bi))
        if bi % step_print == 0 or bi == len(dl_train):
            _print_batch_progress(cons, bi, len(dl_train), loss_sum, correct, total, seen, t0, load_ms, gpu_ms)
        if prog:
            prog.update(epoch_task, advance=1)
        prev_end = _time.perf_counter()
    if prog:
        prog.remove_task(epoch_task)
    train_acc = correct / max(1, total)
    train_loss = (loss_sum / max(1, total)) if total else 0.0
    return train_acc, seen, io_times, gpu_times, batch_times, (_time.perf_counter() - t0), train_loss


def _live_val_step(args, model, device, telemetry: Telemetry | None, cons, ep: int, global_step: int):
    """Run a tiny validation on a shuffled batch and plot immediately.

    Minimal by design: evaluates one batch from a shuffled val loader and logs
    a 'val_live' series to Visdom/MLflow via Telemetry, plus a dim console line.
    """
    try:
        dl = getattr(args, '_dl_val_live', None)
        if dl is None:
            return
        it = getattr(args, '_dl_val_it', None)
        if it is None:
            it = iter(dl); args._dl_val_it = it
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl); args._dl_val_it = it
            xb, yb = next(it)
        import torch as _torch
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        model.eval()
        with _torch.no_grad():
            pr = model(xb).argmax(1)
            acc = float((pr == yb).float().mean().item())
        try:
            cons.print(f"[dim]val_live step={global_step} acc={acc:.3f}[/]")
        except Exception:
            pass
        if telemetry is not None:
            try:
                telemetry.scalar2('val_live', 'val_live', global_step, acc, title='Val (Live)', ylabel='acc', win='vkb_val_live')
                telemetry.metric('val_live', acc, step=global_step)
            except Exception:
                pass
    except Exception:
        pass


def _validate_epoch(dl_val, model, device, n_classes, labels, cons, loss_fn=None):
    import numpy as _np
    if not dl_val:
        return None, None
    cons.print("[dim]Validating...[/]")
    model.eval(); vc, vt = 0, 0; loss_sum = 0.0
    y_true, y_pred = [], []
    import torch as _torch
    with _torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pr = logits.argmax(1)
            if loss_fn is not None:
                try:
                    loss_sum += float(loss_fn(logits, yb).item())
                except Exception:
                    pass
            vt += yb.size(0); vc += int((pr == yb).sum().item())
            y_true.extend(yb.cpu().tolist()); y_pred.extend(pr.cpu().tolist())
    val_acc = vc / max(1, vt)
    cm = _np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    from rich.table import Table as _T
    ctab = _T(title="Confusion (val)")
    ctab.add_column("")
    for l in labels: ctab.add_column(l)
    for i, l in enumerate(labels):
        row = [l] + [str(int(cm[i, j])) for j in range(n_classes)]
        ctab.add_row(*row)
    cons.print(ctab)
    cons.print(f"[dim]confusion_raw={cm.tolist()}[/]")
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec_d = tp + fp; rec_d = tp + fn
        prec = (tp / prec_d) if prec_d > 0 else 0.0
        rec = (tp / rec_d) if rec_d > 0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    mF1 = float(_np.mean(f1s)) if f1s else 0.0
    cons.print(f"[dim]macro_f1={mF1:.3f}[/]")
    val_loss = (loss_sum / max(1, vt)) if vt else None
    return val_acc, val_loss


def _print_perf(io_times, gpu_times, batch_times, cons, bsz: int):
    import numpy as _np
    if not (io_times and gpu_times):
        return
    io_arr = _np.array(io_times)
    gpu_arr = _np.array(gpu_times)
    batch_arr = _np.array(batch_times)
    def pct(a, p):
        i = int(max(0, min(len(a)-1, round(p*(len(a)-1)))))
        return float(_np.sort(a)[i])
    io_avg, io_p90 = float(io_arr.mean()), pct(io_arr, 0.90)
    gpu_avg, gpu_p90 = float(gpu_arr.mean()), pct(gpu_arr, 0.90)
    bat_avg = float(batch_arr.mean())
    thr = (bsz * 1000.0 / bat_avg) if bat_avg > 0 else 0.0  # samples/sec from avg batch time
    stall = (io_avg / (io_avg + gpu_avg)) if (io_avg + gpu_avg) > 0 else 0.0
    cons.print(
        f"[dim]Perf: batches={len(io_arr)} bs={int(bsz)} io_ms(avg/p90)={io_avg:.1f}/{io_p90:.1f} "
        f"gpu_ms(avg/p90)={gpu_avg:.1f}/{gpu_p90:.1f} batch_ms(avg)={bat_avg:.1f} samples/s={thr:.1f} stall={stall:.2f}[/]"
    )

## Visdom helpers moved to vkb.vis (imported above)


def _prepare_data(args, cons):
    vids, labels, lab2i = _labels_and_videos(args.data)
    # Show newest video included (quick sanity check)
    try:
        import os as _os
        latest = max(vids, key=lambda vl: _os.path.getmtime(vl[0])) if vids else None
        if latest:
            cons.print(f"[dim]Newest video: {_os.path.relpath(latest[0])} label={latest[1]}[/]")
    except Exception:
        pass
    # Expose videos for later diagnostics (e.g., cache summary) without relying
    # on worker-local dataset state.
    try:
        args._vids = vids
    except Exception:
        pass
    samples = _build_samples(vids, lab2i, stride=1)
    n_classes = len(labels)
    if not samples or n_classes < 2:
        raise ValueError("need at least 2 classes and some frames to fine‑tune")
    if getattr(args, 'eval_mode', 'tail') == 'tail-per-video':
        tr_idx, va_idx, te_idx = _per_video_tail_split_three(samples, getattr(args,'eval_split',0.0), getattr(args,'test_split',0.0))
    else:
        tr_idx, va_idx, te_idx = _tail_split_three(samples, n_classes, getattr(args,'eval_split',0.0), getattr(args,'test_split',0.0))
    cons.print(f"[dim]Videos: {len(vids)}  Classes: {n_classes}  Frames: {len(samples)}  Train: {len(tr_idx)}  Val: {len(va_idx)}  Test: {len(te_idx)}[/]")
    return vids, labels, samples, n_classes, tr_idx, va_idx, te_idx


def _print_device(cons, device):
    import torch
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            cons.print(f"[dim]Device: cuda ({name}) {props.total_memory/1e9:.1f}GB VRAM[/]")
        else:
            cons.print(f"[dim]Device: {device}[/]")
    except Exception:
        cons.print(f"[dim]Device: {device}[/]")


def _print_hparams(cons, args):
    from rich.table import Table as _T
    hp = _T(title="HParams")
    hp.add_column("key"); hp.add_column("value")
    for k, v in [("backbone", getattr(args, 'backbone', 'mobilenetv3_small_100')), ("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("lr", args.lr), ("weight_decay", args.wd), ("eval_split", args.eval_split),
                 ("eval_mode", args.eval_mode), ("optimizer", "AdamW"), ("label_smoothing", getattr(args, 'label_smoothing', 0.05)),
                 ("drop_path", getattr(args, 'drop_path', 0.0)), ("dropout", getattr(args, 'dropout', 0.0)),
                 ("aug", getattr(args, 'aug', 'none')),
                 ("img_size", getattr(args, 'img_size', 224)),
                 ("brightness", getattr(args, 'brightness', 0.0)), ("warp", getattr(args, 'warp', 0.0)), ("shift", getattr(args, 'shift', 0.0)),
                 ("sat", getattr(args, 'sat', 0.0)), ("contrast", getattr(args, 'contrast', 0.0)), ("wb", getattr(args, 'wb', 0.0)), ("hue", getattr(args, 'hue', 0.0)), ("rot_deg", getattr(args, 'rot_deg', 0.0)),
                 ("val_eq_eps", getattr(args, 'val_eq_eps', VAL_EQ_EPS))]:
        hp.add_row(str(k), str(v))
    cons.print(hp)


def _make_loaders(args, samples, tr_idx, va_idx, n_classes, device, te_idx=None):
    import torch
    from torch.utils.data import DataLoader, Subset
    import os as _os
    dyn = (bool(getattr(args, 'dynamic_aug', False)) and not bool(getattr(args, 'no_dynamic_aug', False)) and not bool(_os.getenv('PYTEST_CURRENT_TEST')))
    z = 0.0 if dyn else None
    img_sz = int(getattr(args, 'img_size', 224) or 224)
    ds_tr = FrameDataset(
        samples, img_size=img_sz, data_root=args.data, mode='train', aug=(getattr(args, 'aug', 'none') if not dyn else 'none'),
        brightness=(z if z is not None else getattr(args, 'brightness', 0.0)),
        warp=(z if z is not None else getattr(args, 'warp', 0.0)),
        shift=(z if z is not None else float(getattr(args, 'shift', 0.0) or 0.0)),
        sat=(z if z is not None else float(getattr(args,'sat',0.0) or 0.0)),
        contrast=(z if z is not None else float(getattr(args,'contrast',0.0) or 0.0)),
        wb=(z if z is not None else float(getattr(args,'wb',0.0) or 0.0)),
        hue=(z if z is not None else float(getattr(args,'hue',0.0) or 0.0)),
        rot_deg=(z if z is not None else float(getattr(args,'rot_deg',0.0) or 0.0)),
        noise_std=float(getattr(args, 'noise_std', 0.0) or 0.0),
        erase_p=float(getattr(args, 'erase_p', 0.0) or 0.0),
        erase_area_min=float(getattr(args, 'erase_area_min', 0.02) or 0.0),
        erase_area_max=float(getattr(args, 'erase_area_max', 0.06) or 0.0)
    )
    ds_va = FrameDataset(samples, img_size=img_sz, data_root=args.data, mode='val', aug='none', brightness=0.0, warp=0.0, shift=0.0, sat=0.0, contrast=0.0, wb=0.0, hue=0.0, rot_deg=0.0, noise_std=0.0, erase_p=0.0)
    val_ds = Subset(ds_va, va_idx) if va_idx else None
    test_ds = Subset(ds_va, te_idx) if te_idx else None
    pin = device.startswith("cuda")
    nw = int(getattr(args, 'workers', 0) or 0)
    try:
        strat = getattr(args, 'sharing_strategy', 'auto')
        if strat == 'auto' and nw > 0:
            torch.multiprocessing.set_sharing_strategy('file_system'); strat = 'file_system'
        elif strat != 'auto':
            torch.multiprocessing.set_sharing_strategy(strat)
    except Exception:
        strat = 'auto'
    pf = int(getattr(args, 'prefetch', 2) or 2)
    no_pw = bool(getattr(args, 'no_persistent_workers', False))
    pw = (bool(getattr(args, 'persistent_workers', False)) and not no_pw) and nw > 0
    sampler = None
    if str(getattr(args, 'class_weights', 'auto')) == 'auto':
        from torch.utils.data import WeightedRandomSampler
        cls_w = _compute_class_weights(samples, tr_idx, n_classes)
        if cls_w:
            sw = [float(cls_w[samples[idx][2]]) for idx in tr_idx]
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    train_ds = Subset(ds_tr, tr_idx)
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                          num_workers=nw, pin_memory=pin, prefetch_factor=(pf if nw>0 else None), persistent_workers=pw)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin,
                        prefetch_factor=(pf if nw>0 else None), persistent_workers=pw) if val_ds else None
    dl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin,
                         prefetch_factor=(pf if nw>0 else None), persistent_workers=pw) if test_ds else None
    # Expose train dataset to callers for dynamic augmentation control
    try:
        args._ds_tr = ds_tr
    except Exception:
        pass
    # Optional shuffled live-validation loader for periodic checks inside an epoch
    try:
        vli = int(getattr(args, 'val_live_interval', 0) or 0)
        if vli > 0 and val_ds is not None:
            from torch.utils.data import DataLoader as _DL
            args._dl_val_live = _DL(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin,
                                    prefetch_factor=(pf if nw>0 else None), persistent_workers=pw)
            args._dl_val_it = None
    except Exception:
        pass
    return dl_train, dl_val, dl_test, (pin, nw, pf, pw, strat)


def _init_model_and_optim(args, n_classes, device, cons, loader_cfg, samples, tr_idx):
    import torch
    from torch import nn
    bkb = getattr(args, 'backbone', 'mobilenetv3_small_100')
    model = _create_timm_model(
        bkb,
        n_classes,
        float(getattr(args, 'dropout', 0.0) or 0.0),
        float(getattr(args, 'drop_path', 0.0) or 0.0),
        pretrained=True,
    )
    model.to(device)
    try:
        torch.backends.cudnn.benchmark = True
        is_cuda = next(model.parameters()).is_cuda
        pin, nw, pf, pw, strat = loader_cfg
        cons.print(f"[dim]model_on_cuda={is_cuda} pin_memory={pin} workers={nw} prefetch={pf if nw>0 else 0} persistent={pw} sharing={strat}[/]")
    except Exception:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    cw = None
    if str(getattr(args, 'class_weights', 'auto')) == 'auto':
        w = _compute_class_weights(samples, tr_idx, n_classes)
        if w:
            import torch as _torch
            cw = _torch.tensor(w, dtype=_torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(getattr(args, 'label_smoothing', 0.05) or 0.0), weight=cw)
    return model, opt, loss_fn


def _save_artifacts(cons, args, model, labels, tr_idx, n_classes, best_val, device, mlflow=None, best_epoch=None, test_acc=None):
    from rich.table import Table
    from vkb.artifacts import save_model
    bundle = {
        "clf_name": "finetune",
        "labels": labels,
        "model_name": getattr(args, 'backbone', 'mobilenetv3_small_100'),
        "state_dict": model.state_dict(),
        "input_size": int(getattr(args, 'img_size', 224) or 224),
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    parts = ["finetune", getattr(args, 'backbone', 'mobilenetv3_small_100')]
    if best_val is not None:
        parts.append(f"val{float(best_val):.3f}")
    if best_epoch is not None:
        parts.append(f"ep{int(best_epoch):02d}")
    path = save_model(bundle, parts, base_dir="models")
    from vkb.artifacts import save_sidecar as _save_sidecar
    meta_path = _save_sidecar(path, {
        "clf_name": "finetune", "model_name": getattr(args, 'backbone', 'mobilenetv3_small_100'), "labels": labels,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.wd,
        "label_smoothing": float(getattr(args, 'label_smoothing', 0.05) or 0.0),
        "drop_path": float(getattr(args, 'drop_path', 0.0) or 0.0), "dropout": float(getattr(args, 'dropout', 0.0) or 0.0),
        "eval_split": args.eval_split, "eval_mode": args.eval_mode, "val_acc": best_val,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "test_acc": float(test_acc) if test_acc is not None else None,
        "aug": str(getattr(args, 'aug', 'none')), "brightness": float(getattr(args, 'brightness', 0.0) or 0.0),
        "warp": float(getattr(args, 'warp', 0.0) or 0.0), "shift": float(getattr(args, 'shift', 0.0) or 0.0),
        "sat": float(getattr(args, 'sat', 0.0) or 0.0), "contrast": float(getattr(args, 'contrast', 0.0) or 0.0), "wb": float(getattr(args, 'wb', 0.0) or 0.0), "hue": float(getattr(args, 'hue', 0.0) or 0.0),
        "device": str(device),
    })
    if mlflow is not None:
        try:
            mlflow.log_artifact(path)
            mlflow.log_artifact(meta_path)
            mlflow.end_run()
        except Exception:
            pass
    t = Table(title="Training Summary (DL)")
    t.add_column("classifier"); t.add_column("backbone"); t.add_column("frames"); t.add_column("classes"); t.add_column("val_acc"); t.add_column("saved")
    t.add_row("finetune", getattr(args, 'backbone', 'mobilenetv3_small_100'), str(len(tr_idx)), str(n_classes), f"{best_val:.3f}" if best_val is not None else "-", path)
    cons.print(t)
    try:
        cons.print(f"Saved model: {path}")
    except Exception:
        pass
    return path


def finetune(args):
    import time
    import numpy as np
    import torch
    from rich.console import Console

    cons = Console()
    # Ensure a unified config is available (optional)
    try:
        from vkb.config import make_config
        args.cfg = getattr(args, 'cfg', None) or make_config(args)
    except Exception:
        pass
    mlflow = _setup_mlflow(args)
    device = _device_from_arg(args.device)
    _assert_device_flags(args, device)
    vids, labels, samples, n_classes, tr_idx, va_idx, te_idx = _prepare_data(args, cons)
    _print_device(cons, device)
    _print_hparams(cons, args)
    _mlflow_log_params(args, mlflow)
    if bool(getattr(args, 'amp', False)):
        if not (device.startswith('cuda') and torch.cuda.is_available()):
            raise RuntimeError("--amp requires CUDA (`--device cuda`).")

    # If DSPy aug is requested, disable our built‑in dynamic aug and set up predictor.
    if bool(getattr(args, 'dspy_aug', False)):
        try: args.no_dynamic_aug = True
        except Exception: pass
    dl_train, dl_val, dl_test, loader_cfg = _make_loaders(args, samples, tr_idx, va_idx, n_classes, device, te_idx=te_idx)
    dyn_enabled = (bool(getattr(args, 'dynamic_aug', False)) and not bool(getattr(args, 'no_dynamic_aug', False)))
    dyn = AugScheduler(getattr(args, '_ds_tr', None), args) if dyn_enabled and getattr(args, '_ds_tr', None) is not None else None
    # Optional DSPy predictor
    pred = None
    dspy_hist = []  # collect {step, strengths..., val_acc}
    if bool(getattr(args, 'dspy_aug', False)):
        try:
            from dspy_agents import AsyncAugPredictor, configure_deepseek, configure_openrouter
            # Configure LM only when requested and just once per run.
            if bool(getattr(args, 'dspy_openrouter', False)):
                configure_openrouter(reasoning_effort=getattr(args, 'dspy_reasoning_effort', None))
                try:
                    cons.print(f"[dim]DSPy: api=openrouter reasoning={getattr(args,'dspy_reasoning_effort',None)}[/]")
                except Exception:
                    pass
            else:
                configure_deepseek()
                try:
                    cons.print("[dim]DSPy: api=deepseek[/]")
                except Exception:
                    pass
            pred = AsyncAugPredictor()
            # seed history with current strengths
            ds = getattr(args, '_ds_tr', None)
            if ds is not None:
                dspy_hist.append({
                    'step': 0,
                    'brightness': float(getattr(ds,'brightness',0.0) or 0.0),
                    'warp': float(getattr(ds,'warp',0.0) or 0.0),
                    'sat': float(getattr(ds,'sat',0.0) or 0.0),
                    'contrast': float(getattr(ds,'contrast',0.0) or 0.0),
                    'hue': float(getattr(ds,'hue',0.0) or 0.0),
                    'wb': float(getattr(ds,'wb',0.0) or 0.0),
                    'rot_deg': float(getattr(ds,'rot_deg',0.0) or 0.0),
                    'val_acc': None,
                })
        except Exception as _e:
            # Surface errors plainly (no fallback)
            cons.print(f"[red]DSPy aug init failed: {_e}[/]")
            raise
    pred_t0 = None
    # Optional OpenRouter reasoning streaming (flag + openrouter)
    reason_thr = None
    model, opt, loss_fn = _init_model_and_optim(args, n_classes, device, cons, loader_cfg, samples, tr_idx)

    import copy as _copy
    best_val = None; best_epoch = None; best_state = None
    ep_times = []
    total_seen = 0
    prog = _init_progress(args)
    viz, viz_wins, telemetry = _setup_viz_and_telemetry(args, cons, mlflow)
    if prog:
        with prog:
            outer = prog.add_task("Epochs", total=args.epochs)
            for ep in range(1, args.epochs + 1):
                # submit next‑epoch suggestion job at epoch start
                if pred is not None:
                    try:
                        import time as _time
                        pred.submit(list(dspy_hist))
                        pred_t0 = _time.perf_counter()
                    except Exception as _e:
                        cons.print(f"[red]DSPy submit failed: {_e}[/]"); raise
                # Kick off reasoning streaming if requested (OpenRouter only)
                if bool(getattr(args, 'dspy_stream_reasoning', False)) and bool(getattr(args, 'dspy_openrouter', False)) and reason_thr is None:
                    try:
                        from dspy_agents import stream_reasoning_openrouter, AugStep
                        model_slug = str(getattr(args, 'dspy_model', 'deepseek/deepseek-reasoner'))
                        effort = getattr(args, 'dspy_reasoning_effort', None)
                        steps = []
                        for h in dspy_hist:
                            try:
                                steps.append(AugStep(**{k: h.get(k) for k in ('step','brightness','warp','shift','sat','contrast','hue','wb','rot_deg','train_acc','train_loss','val_acc','val_loss')}))
                            except Exception:
                                pass
                        def _run_stream():
                            stream_reasoning_openrouter(steps, model=model_slug, effort=effort, on_delta=lambda s: cons.print(f"[dim]DSPy reasoning: {s}", soft_wrap=True))
                        import threading as _th
                        reason_thr = _th.Thread(target=_run_stream, daemon=True)
                        reason_thr.start()
                    except Exception as _e:
                        cons.print(f"[yellow]DSPy reasoning stream unavailable: {_e}[/]")
                _ret = _train_epoch(ep, args, dl_train, model, device, opt, loss_fn, cons, prog, viz, viz_wins, mlflow=mlflow, telemetry=telemetry)
                if isinstance(_ret, tuple) and len(_ret) >= 7:
                    tr_acc, seen, io_t, gpu_t, bat_t, ep_dt, tr_loss = _ret
                else:
                    tr_acc, seen, io_t, gpu_t, bat_t, ep_dt = _ret
                    tr_loss = 0.0
                msg = f"epoch {ep}/{args.epochs} train_acc={tr_acc:.3f}"
                _v = _validate_epoch(dl_val, model, device, n_classes, labels, cons, loss_fn=loss_fn)
                if isinstance(_v, tuple):
                    val_acc, val_loss = _v
                else:
                    val_acc, val_loss = _v, None
                improved_now = False
                equal_now = False
                if val_acc is not None:
                    msg += f" val_acc={val_acc:.3f}"
                    eps = float(getattr(args, 'val_eq_eps', VAL_EQ_EPS) or VAL_EQ_EPS)
                    improved_now, equal_now = _val_compare(val_acc, best_val, eps=eps)
                    if improved_now:
                        best_val = val_acc; best_epoch = ep; best_state = _copy.deepcopy(model.state_dict())
                    elif equal_now:
                        try:
                            cons.print(f"[yellow]Tie within ±{float(eps):.3f}: keeping aug change.[/]")
                        except Exception:
                            pass
                cons.print(msg)
                if mlflow is not None:
                    try:
                        mlflow.log_metric('train_acc', float(tr_acc), step=ep)
                        if val_acc is not None:
                            mlflow.log_metric('val_acc', float(val_acc), step=ep)
                    except Exception:
                        pass
                # Print policy used this epoch (aug strengths + optional reg knobs) with metrics
                try:
                    ds = getattr(args, '_ds_tr', None)
                    if ds is not None:
                        do = float(getattr(args, 'dropout', 0.0) or 0.0)
                        dp = float(getattr(args, 'drop_path', 0.0) or 0.0)
                        pol_line = (
                            f"Policy used: "
                            f"brightness={float(getattr(ds,'brightness',0.0) or 0.0):.3f} "
                            f"warp={float(getattr(ds,'warp',0.0) or 0.0):.3f} "
                            f"shift={float(getattr(ds,'shift',0.0) or 0.0):.3f} "
                            f"saturation={float(getattr(ds,'sat',0.0) or 0.0):.3f} "
                            f"contrast={float(getattr(ds,'contrast',0.0) or 0.0):.3f} "
                            f"hue={float(getattr(ds,'hue',0.0) or 0.0):.3f} "
                            f"white_balance={float(getattr(ds,'wb',0.0) or 0.0):.3f} "
                            f"rotation_deg={float(getattr(ds,'rot_deg',0.0) or 0.0):.1f} "
                            f"dropout={do:.3f} drop_path={dp:.3f} "
                        )
                        try:
                            tl = float(tr_loss)
                        except Exception:
                            tl = 0.0
                        vl = float(val_loss) if (val_loss is not None) else 0.0
                        va = float(val_acc) if (val_acc is not None) else 0.0
                        cons.print(f"[dim]{pol_line}train: loss={tl:.3f} acc={tr_acc:.3f} | val: loss={vl:.3f} acc={va:.3f}[/]")
                except Exception:
                    pass
                _epoch_after(cons, telemetry, ep, args, tr_acc, val_acc)
                ep_times.append(ep_dt); total_seen += seen
                _print_perf(io_t, gpu_t, bat_t, cons, int(getattr(args, 'batch_size', 0) or 0))
                # record epoch for DSPy history
                if pred is not None and getattr(args, '_ds_tr', None) is not None:
                    ds = args._ds_tr
                    dspy_hist.append({
                        'step': ep,
                        'brightness': float(getattr(ds,'brightness',0.0) or 0.0),
                        'warp': float(getattr(ds,'warp',0.0) or 0.0),
                        'shift': float(getattr(ds,'shift',0.0) or 0.0),
                        'sat': float(getattr(ds,'sat',0.0) or 0.0),
                        'contrast': float(getattr(ds,'contrast',0.0) or 0.0),
                        'hue': float(getattr(ds,'hue',0.0) or 0.0),
                        'wb': float(getattr(ds,'wb',0.0) or 0.0),
                        'rot_deg': float(getattr(ds,'rot_deg',0.0) or 0.0),
                        'dropout': float(getattr(args,'dropout',0.0) or 0.0),
                        'drop_path': float(getattr(args,'drop_path',0.0) or 0.0),
                        'train_acc': float(tr_acc),
                        'train_loss': float(tr_loss),
                        'val_acc': (float(val_acc) if val_acc is not None else None),
                        'val_loss': (float(val_loss) if val_loss is not None else None),
                    })
                if dyn is not None:
                    dyn.update(bool(improved_now), equal=bool(equal_now))
                # apply DSPy suggestion if ready (non‑blocking)
                if pred is not None and getattr(args, '_ds_tr', None) is not None and pred.ready():
                    try:
                        sugg = pred.result(timeout=0.0)
                        ds = args._ds_tr
                        for k, v in sugg.items():
                            if hasattr(ds, k):
                                setattr(ds, k, float(v))
                        # ensure warp reaches the Augment object
                        try:
                            if hasattr(ds, '_augment') and hasattr(ds._augment, 'warp'):
                                ds._augment.warp = float(sugg.get('warp', getattr(ds,'warp',0.0)))
                        except Exception:
                            pass
                        try:
                            import time as _time
                            dt = (_time.perf_counter() - pred_t0) if pred_t0 is not None else 0.0
                        except Exception:
                            dt = 0.0
                        try:
                            parts = [
                                f"brightness={float(sugg.get('brightness',0.0)):.3f}",
                                f"warp={float(sugg.get('warp',0.0)):.3f}",
                                f"shift={float(sugg.get('shift',0.0)):.3f}",
                                f"saturation={float(sugg.get('sat',0.0)):.3f}",
                                f"contrast={float(sugg.get('contrast',0.0)):.3f}",
                                f"hue={float(sugg.get('hue',0.0)):.3f}",
                                f"white_balance={float(sugg.get('wb',0.0)):.3f}",
                                f"rotation_deg={float(sugg.get('rot_deg',0.0)):.1f}",
                            ]
                            if 'dropout' in sugg and sugg['dropout'] is not None:
                                parts.append(f"dropout={float(sugg['dropout']):.3f}")
                            if 'drop_path' in sugg and sugg['drop_path'] is not None:
                                parts.append(f"drop_path={float(sugg['drop_path']):.3f}")
                            msg_vals = " ".join(parts)
                        except Exception:
                            msg_vals = ""
                        cons.print(f"[dim]DSPy aug applied for next epoch in {dt:.2f}s: {msg_vals}[/]")
                    except TimeoutError:
                        pass
                    except Exception as _e:
                        cons.print(f"[red]DSPy result failed: {_e}[/]")
                _save_epoch_model(cons, args, model, labels, ep)
                prog.advance(outer, 1)
    else:
        for ep in range(1, args.epochs + 1):
            if pred is not None:
                try:
                    import time as _time
                    pred.submit(list(dspy_hist))
                    pred_t0 = _time.perf_counter()
                except Exception as _e:
                    cons.print(f"[red]DSPy submit failed: {_e}[/]"); raise
            # Kick off reasoning streaming if requested (OpenRouter only)
            if bool(getattr(args, 'dspy_stream_reasoning', False)) and bool(getattr(args, 'dspy_openrouter', False)) and reason_thr is None:
                try:
                    from dspy_agents import stream_reasoning_openrouter, AugStep
                    model_slug = str(getattr(args, 'dspy_model', 'deepseek/deepseek-reasoner'))
                    effort = getattr(args, 'dspy_reasoning_effort', None)
                    steps = []
                    for h in dspy_hist:
                        try:
                            steps.append(AugStep(**{k: h.get(k) for k in ('step','brightness','warp','shift','sat','contrast','hue','wb','rot_deg','train_acc','train_loss','val_acc','val_loss')}))
                        except Exception:
                            pass
                    def _run_stream():
                        stream_reasoning_openrouter(steps, model=model_slug, effort=effort, on_delta=lambda s: cons.print(f"[dim]DSPy reasoning: {s}", soft_wrap=True))
                    import threading as _th
                    reason_thr = _th.Thread(target=_run_stream, daemon=True)
                    reason_thr.start()
                except Exception as _e:
                    cons.print(f"[yellow]DSPy reasoning stream unavailable: {_e}[/]")
            _ret = _train_epoch(ep, args, dl_train, model, device, opt, loss_fn, cons, prog, viz, viz_wins, mlflow=mlflow, telemetry=telemetry)
            if isinstance(_ret, tuple) and len(_ret) >= 7:
                tr_acc, seen, io_t, gpu_t, bat_t, ep_dt, tr_loss = _ret
            else:
                tr_acc, seen, io_t, gpu_t, bat_t, ep_dt = _ret
                tr_loss = 0.0
            msg = f"epoch {ep}/{args.epochs} train_acc={tr_acc:.3f}"
            _v = _validate_epoch(dl_val, model, device, n_classes, labels, cons, loss_fn=loss_fn)
            if isinstance(_v, tuple):
                val_acc, val_loss = _v
            else:
                val_acc, val_loss = _v, None
            improved_now = False
            equal_now = False
            if val_acc is not None:
                msg += f" val_acc={val_acc:.3f}"
                eps = float(getattr(args, 'val_eq_eps', VAL_EQ_EPS) or VAL_EQ_EPS)
                improved_now, equal_now = _val_compare(val_acc, best_val, eps=eps)
                if improved_now:
                    best_val = val_acc; best_epoch = ep; best_state = _copy.deepcopy(model.state_dict())
                elif equal_now:
                    try:
                        cons.print(f"[yellow]Tie within ±{float(eps):.3f}: keeping aug change.[/]")
                    except Exception:
                        pass
            cons.print(msg)
            try:
                ds = getattr(args, '_ds_tr', None)
                if ds is not None:
                    do = float(getattr(args, 'dropout', 0.0) or 0.0)
                    dp = float(getattr(args, 'drop_path', 0.0) or 0.0)
                    pol_line = (
                        f"Policy used: "
                        f"brightness={float(getattr(ds,'brightness',0.0) or 0.0):.3f} "
                        f"warp={float(getattr(ds,'warp',0.0) or 0.0):.3f} "
                        f"saturation={float(getattr(ds,'sat',0.0) or 0.0):.3f} "
                        f"contrast={float(getattr(ds,'contrast',0.0) or 0.0):.3f} "
                        f"hue={float(getattr(ds,'hue',0.0) or 0.0):.3f} "
                        f"white_balance={float(getattr(ds,'wb',0.0) or 0.0):.3f} "
                        f"rotation_deg={float(getattr(ds,'rot_deg',0.0) or 0.0):.1f} "
                        f"dropout={do:.3f} drop_path={dp:.3f} "
                    )
                    try:
                        tl = float(tr_loss)
                    except Exception:
                        tl = 0.0
                    vl = float(val_loss) if (val_loss is not None) else 0.0
                    va = float(val_acc) if (val_acc is not None) else 0.0
                    cons.print(f"[dim]{pol_line}train: loss={tl:.3f} acc={tr_acc:.3f} | val: loss={vl:.3f} acc={va:.3f}[/]")
            except Exception:
                pass
            _epoch_after(cons, telemetry, ep, args, tr_acc, val_acc)
            ep_times.append(ep_dt); total_seen += seen
            _print_perf(io_t, gpu_t, bat_t, cons, int(getattr(args, 'batch_size', 0) or 0))
            if pred is not None and getattr(args, '_ds_tr', None) is not None:
                ds = args._ds_tr
                dspy_hist.append({
                    'step': ep,
                    'brightness': float(getattr(ds,'brightness',0.0) or 0.0),
                    'warp': float(getattr(ds,'warp',0.0) or 0.0),
                    'sat': float(getattr(ds,'sat',0.0) or 0.0),
                    'contrast': float(getattr(ds,'contrast',0.0) or 0.0),
                    'hue': float(getattr(ds,'hue',0.0) or 0.0),
                    'wb': float(getattr(ds,'wb',0.0) or 0.0),
                    'rot_deg': float(getattr(ds,'rot_deg',0.0) or 0.0),
                    'dropout': float(getattr(args,'dropout',0.0) or 0.0),
                    'drop_path': float(getattr(args,'drop_path',0.0) or 0.0),
                    'train_acc': float(tr_acc),
                    'train_loss': float(tr_loss),
                    'val_acc': (float(val_acc) if val_acc is not None else None),
                    'val_loss': (float(val_loss) if val_loss is not None else None),
                })
            if dyn is not None:
                dyn.update(bool(improved_now), equal=bool(equal_now))
            if pred is not None and getattr(args, '_ds_tr', None) is not None and pred.ready():
                try:
                    sugg = pred.result(timeout=0.0)
                    ds = args._ds_tr
                    for k, v in sugg.items():
                        if hasattr(ds, k):
                            setattr(ds, k, float(v))
                    try:
                        if hasattr(ds, '_augment') and hasattr(ds._augment, 'warp'):
                            ds._augment.warp = float(sugg.get('warp', getattr(ds,'warp',0.0)))
                    except Exception:
                        pass
                    # Apply training knobs (dropout/drop_path) if present
                    try:
                        import torch.nn as _nn
                        if 'dropout' in sugg and sugg['dropout'] is not None:
                            dval = float(sugg['dropout'])
                            setattr(args, 'dropout', dval)
                            for m in model.modules():
                                if isinstance(m, _nn.Dropout):
                                    m.p = dval
                        if 'drop_path' in sugg and sugg['drop_path'] is not None:
                            dpval = float(sugg['drop_path'])
                            setattr(args, 'drop_path', dpval)
                            for m in model.modules():
                                if hasattr(m, 'drop_prob'):
                                    try: m.drop_prob = dpval
                                    except Exception: pass
                    except Exception:
                        pass
                    try:
                        import time as _time
                        dt = (_time.perf_counter() - pred_t0) if pred_t0 is not None else 0.0
                    except Exception:
                        dt = 0.0
                    try:
                        parts = [
                            f"brightness={float(sugg.get('brightness',0.0)):.3f}",
                            f"warp={float(sugg.get('warp',0.0)):.3f}",
                            f"saturation={float(sugg.get('sat',0.0)):.3f}",
                            f"contrast={float(sugg.get('contrast',0.0)):.3f}",
                            f"hue={float(sugg.get('hue',0.0)):.3f}",
                            f"white_balance={float(sugg.get('wb',0.0)):.3f}",
                            f"rotation_deg={float(sugg.get('rot_deg',0.0)):.1f}",
                        ]
                        if 'dropout' in sugg and sugg['dropout'] is not None:
                            parts.append(f"dropout={float(sugg['dropout']):.3f}")
                        if 'drop_path' in sugg and sugg['drop_path'] is not None:
                            parts.append(f"drop_path={float(sugg['drop_path']):.3f}")
                        msg_vals = " ".join(parts)
                    except Exception:
                        msg_vals = ""
                    cons.print(f"[dim]DSPy aug applied for next epoch in {dt:.2f}s: {msg_vals}[/]")
                except TimeoutError:
                    pass
                except Exception as _e:
                    cons.print(f"[red]DSPy result failed: {_e}[/]")
            _save_epoch_model(cons, args, model, labels, ep)

    # Load best epoch weights if captured
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass
    _print_summary(ep_times, total_seen, cons)
    # Optional test eval using best model
    test_acc = None
    if dl_test is not None:
        try:
            test_acc, _ = _validate_epoch(dl_test, model, device, n_classes, labels, cons, loss_fn=loss_fn)
            if mlflow is not None and test_acc is not None:
                try: mlflow.log_metric('test_acc', float(test_acc), step=int(best_epoch or 0))
                except Exception: pass
        except Exception:
            test_acc = None
    return _save_artifacts(cons, args, model, labels, tr_idx, n_classes, best_val, device, mlflow=mlflow, best_epoch=best_epoch, test_acc=test_acc)
