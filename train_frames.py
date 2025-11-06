#!/usr/bin/env python3
import argparse
from vkb.io import list_videos
from vkb.emb import create_embedder
from vkb.artifacts import save_model, save_sidecar
from vkb.cache import cache_dir as _cache_dir, cache_path_for as _cache_path_for, load_embeddings as _cache_load, save_embeddings as _cache_save


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Embed frames and train a frame classifier")
    p.add_argument("--data", default="data")
    p.add_argument("--embed-model", dest="embed_model", default="mobilenetv3_small_100", help="Classic path: embedding backbone for ridge/xgb/logreg")
    p.add_argument("--backbone", dest="backbone", default="mobilenetv3_small_100", help="DL path: timm backbone for fine‑tune (separate from --embed-model)")
    p.add_argument("--clf", choices=["xgb","ridge","logreg","dl","mp_logreg"], default="xgb")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    p.add_argument("--hpo-alpha", type=int, default=10, help="Ridge HPO iterations (log-uniform alpha)")
    p.add_argument("--hpo-xgb", type=int, default=10, help="XGBoost HPO trials (random search)")
    p.add_argument("--C", type=float, default=1.0, help="LogisticRegression inverse regularization (C)")
    p.add_argument("--hpo-logreg", type=int, default=10, help="LogReg HPO trials (log-uniform C)")
    p.add_argument("--logreg-max-iter", type=int, default=500, help="LogReg max iterations (increase to avoid convergence warnings)")
    p.add_argument("--eval-split", type=float, default=0.01, help="validation split fraction (0 to disable)")
    p.add_argument("--eval-mode", choices=["random","tail","tail-per-video"], default="tail", help="random stratified vs per-class chronological tail split; 'tail-per-video' uses the tail of each video")
    p.add_argument("--test-split", type=float, default=0.0, help="optional tail test holdout fraction (per-class); evaluated once at end")
    # Fine-tune (dl) specific
    p.add_argument("--epochs", type=int, default=10, help="DL: epochs")
    p.add_argument("--batch-size", type=int, default=128, help="DL: batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="DL: learning rate")
    p.add_argument("--wd", type=float, default=1e-4, help="DL: weight decay")
    p.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=0.05, help="DL: CrossEntropy label smoothing (default 0.05)")
    p.add_argument("--drop-path", type=float, default=0.25, help="DL: drop path (stochastic depth) rate; heavy by default")
    p.add_argument("--dropout", type=float, default=0.0, help="DL: classifier dropout (if model supports it)")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto", help="DL: device")
    p.add_argument("--require-cuda", action="store_true", help="DL: fail if not running on CUDA")
    p.add_argument("--workers", type=int, default=1, help="DL: dataloader workers (default 1)")
    p.add_argument("--rich-progress", action="store_true", help="DL: show rich progress bars for epochs/batches")
    p.add_argument("--prefetch", type=int, default=1, help="DL: prefetch_factor when workers>0")
    p.add_argument("--persistent-workers", action="store_true", default=False, help="DL: persistent_workers in DataLoader (default OFF)")
    p.add_argument("--no-persistent-workers", action="store_true", help="DL: disable persistent workers")
    p.add_argument("--sharing-strategy", choices=["auto","file_system","file_descriptor"], default="auto", help="DL: torch.multiprocessing sharing strategy")
    # MediaPipe landmarks mode (mp_logreg)
    p.add_argument("--mp-stride", type=int, default=5, help="MP: process every Nth frame per video (default 5)")
    p.add_argument("--mp-max-frames", type=int, default=200, help="MP: max frames per video to use (default 200)")
    # System stability / diagnostics
    p.add_argument("--cap-threads", action="store_true", help="Cap BLAS/OpenCV/Torch threads to 1 to reduce UI freezes")
    p.add_argument("--nice", type=int, default=10, help="Positive niceness to lower CPU priority (Linux only)")
    p.add_argument("--diag-system", action="store_true", help="Print lightweight system diagnostics each epoch (loadavg, threads)")
    p.add_argument("--aug", choices=["none","light","heavy","rot360"], default="rot360", help="DL: augmentation policy (zoom-out/rotation)")
    p.add_argument("--noise-std", type=float, default=0.05, help="DL: per-sample Gaussian noise; std ~ U(0, noise_std). 0 disables (train only)")
    p.add_argument("--brightness", type=float, default=0.25, help="DL: brightness jitter amplitude b (applies factor in [1-b,1+b], train only)")
    p.add_argument("--sat", type=float, default=0.40, help="DL: saturation jitter amplitude s (scales distance from gray in [1-s,1+s], train only)")
    p.add_argument("--contrast", type=float, default=0.15, help="DL: contrast jitter amplitude c (scales distance from mean in [1-c,1+c], train only)")
    p.add_argument("--wb", type=float, default=0.30, help="DL: per-channel white-balance jitter amplitude (per-channel scales in [1-wb,1+wb], train only)")
    p.add_argument("--hue", type=float, default=0.30, help="DL: hue shift amplitude h (~radians/pi; rotates chroma by ±h·π), train only")
    p.add_argument("--shift", type=float, default=0.25, help="DL: heavy roll-around XY shift (fraction of size, e.g., 0.25≈±25%%), train only")
    # Resolution
    p.add_argument("--img-size", dest="img_size", type=int, default=224, help="DL: input resolution (square); default 224")
    # Random erasing (train only)
    p.add_argument("--erase-p", dest="erase_p", type=float, default=0.3, help="DL: random erasing probability per sample (train only); 0 disables (default 0.3)")
    p.add_argument("--erase-area-min", dest="erase_area_min", type=float, default=0.02, help="DL: min erased area fraction")
    p.add_argument("--erase-area-max", dest="erase_area_max", type=float, default=0.06, help="DL: max erased area fraction")
    p.add_argument("--rot-deg", dest="rot_deg", type=float, default=360.0, help="DL: rotation magnitude in degrees (±rot_deg), train only; used by dynamic aug too (default full rotation)")
    # DSPy aug suggestions (off by default). When enabled, we fetch next‑epoch
    # aug strengths via DeepSeek/DSPy while the epoch runs and apply them at epoch end.
    p.add_argument("--dspy-aug", action="store_true", help="DL: enable DSPy (DeepSeek) aug suggestion per epoch")
    p.add_argument("--dspy-openrouter", action="store_true", help="DL: route DSPy aug via OpenRouter API instead of DeepSeek API")
    p.add_argument("--dspy-reasoning-effort", choices=["low","medium","high"], default="low", help="DL: (OpenRouter) reasoning effort; default=low")
    p.add_argument("--dspy-stream-reasoning", action="store_true", help="DL: (OpenRouter only) stream reasoning text and print it live")
    p.add_argument("--dspy-model", default="deepseek/deepseek-reasoner", help="DL: (OpenRouter) model to use for DSPy reasoning stream")
    # Dynamic aug is OFF by default now. Keep --dynamic-aug to opt in and --no-dynamic-aug for symmetry.
    p.add_argument("--dynamic-aug", dest="dynamic_aug", action="store_true", help="DL: enable adaptive augmentation controller (off by default)")
    p.add_argument("--no-dynamic-aug", dest="no_dynamic_aug", action="store_true", help="DL: disable adaptive augmentation (use static strengths)")
    p.add_argument("--warp", type=float, default=0.30, help="DL: perspective warp strength in [0,1]; 0 disables (train only)")
    p.add_argument("--class-weights", choices=["none","auto"], default="none", help="Class imbalance handling: 'auto' uses weights/sampler (DL) or sample_weight (classic)")
    p.add_argument("--allow-test-labels", action="store_true", help="Allow training when labels are exactly ['A','B'] (guardrail for real training)")
    # Visdom logging of augmented images (DL only)
    p.add_argument("--visdom-aug", type=int, default=4, help="DL: log this many augmented images from the first batch of each epoch; 0 disables")
    p.add_argument("--visdom-metrics", action="store_true", help="DL: plot train/val accuracy lines in Visdom (uses same env/port)")
    p.add_argument("--visdom-env", type=str, default="vkb-aug", help="DL: Visdom env (not default)")
    p.add_argument("--visdom-port", type=int, default=8097, help="DL: Visdom port")
    # Validation/equality epsilon (DL)
    p.add_argument("--val-eq-eps", type=float, default=0.002, help="DL: epsilon for considering val_acc equal (ties keep aug changes)")
    # Live validation during epoch (DL)
    p.add_argument("--val-live-interval", type=int, default=0, help="DL: run a quick shuffled val every N train batches; 0 disables")
    # MLflow logging
    p.add_argument("--mlflow", action="store_true", help="Log run to MLflow (requires mlflow installed)")
    p.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (defaults to env if unset)")
    p.add_argument("--mlflow-exp", type=str, default="vkb", help="MLflow experiment name")
    p.add_argument("--mlflow-run-name", type=str, default=None, help="Optional MLflow run name")
    # Output dirs
    p.add_argument("--models-dir", type=str, default=None, help="Override models output directory (defaults to env VKB_MODELS_DIR or ./models)")
    p.set_defaults(dynamic_aug=False)
    return p.parse_args(argv)


def _clf_display_name(clf_flag: str) -> str:
    return "xgboost" if clf_flag == "xgb" else clf_flag


def _hpo_ridge(X, y, iters: int, seed: int = 0, logger=None, idx_by_class=None, eval_frac: float | None = None):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.default_rng(seed)

    # Build train/val split
    if idx_by_class and eval_frac is not None and 0.0 < eval_frac < 0.9:
        # chronological tail per class using provided indices
        yarr = np.asarray(y)
        tr_idx, va_idx = [], []
        n_classes = len(idx_by_class)
        for ci in range(n_classes):
            cls_idx = idx_by_class.get(ci, [])
            ccount = len(cls_idx)
            if ccount < 2:
                raise ValueError("HPO needs >=2 samples per class for tail split")
            val_n = max(1, int(round(ccount * eval_frac)))
            val_n = min(ccount - 1, val_n)
            tr_idx.extend(cls_idx[:-val_n])
            va_idx.extend(cls_idx[-val_n:])
        Xtr, ytr = np.asarray(X)[tr_idx], yarr[tr_idx]
        Xva, yva = np.asarray(X)[va_idx], yarr[va_idx]
    else:
        # fallback: random stratified
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    best_a, best_s = None, -1.0
    trials = []
    for i in range(max(1, iters)):
        a = float(10 ** rng.uniform(-4, 2))
        clf = RidgeClassifier(alpha=a).fit(Xtr, ytr)
        s = float(clf.score(Xva, yva))
        trials.append((a, s))
        if logger:
            try:
                logger(i + 1, a, s)
            except Exception:
                pass
        if s > best_s:
            best_a, best_s = a, s
    return best_a, best_s, trials


def _split_tail_indices(idx_by_class, frac: float):
    tr_idx, va_idx = [], []
    for ci, cls_idx in idx_by_class.items():
        ccount = len(cls_idx)
        val_n = max(1, int(round(ccount * frac)))
        val_n = min(ccount - 1, val_n)
        tr_idx.extend(cls_idx[:-val_n])
        va_idx.extend(cls_idx[-val_n:])
    return tr_idx, va_idx

def _split_tail_indices_three(idx_by_class, val_frac: float, test_frac: float):
    # Per-class chronological: train | val | test (tail). Fractions are of total per-class count.
    tr_idx, va_idx, te_idx = [], [], []
    for ci, cls_idx in idx_by_class.items():
        ccount = len(cls_idx)
        if ccount == 0:
            continue
        tv = max(0, int(round(ccount * test_frac)))
        vv = max(0, int(round(ccount * val_frac)))
        # ensure at least 1 train if possible when there are frames
        total_tail = min(ccount, tv + vv)
        head = max(0, ccount - total_tail)
        # assign slices
        tr_idx.extend(cls_idx[:head])
        if vv > 0:
            va_idx.extend(cls_idx[head:head+vv])
        if tv > 0:
            te_idx.extend(cls_idx[head+vv:head+vv+tv])
    return tr_idx, va_idx, te_idx

def _split_tail_per_video_slices(video_slices, val_frac: float, test_frac: float):
    """Per‑video chronological split with global fractions.

    - Allocates total val/test frames across videos proportionally to video length
      (Hamilton rounding), so sum(val)≈eval_frac·total_frames and sum(test)≈test_frac·total_frames.
    - Within each video: train=head, val=middle, test=tail.
    - We reserve one train frame per video when possible (val capped by capacity).
    """
    vids = list(video_slices)
    lens = [max(0, int(e) - int(s)) for (s, e) in vids]
    T = sum(lens)
    if T == 0:
        return [], [], []
    # Hamilton apportionment helper
    def _apportion(counts, frac, cap=None):
        import math
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

    # First apportion test from tails
    test_counts = _apportion(lens, test_frac)
    # Then apportion val within remaining capacity (reserve 1 train when possible)
    cap_for_val = [max(0, lens[i] - test_counts[i] - 1) for i in range(len(lens))]
    val_counts = _apportion(lens, val_frac, cap=cap_for_val)
    tr_idx, va_idx, te_idx = [], [], []
    for (start, end), c, vv, tv in zip(vids, lens, val_counts, test_counts):
        if c <= 0:
            continue
        head = max(0, c - (vv + tv))
        if head > 0:
            tr_idx.extend(range(start, start + head))
        if vv > 0:
            va_idx.extend(range(start + head, start + head + vv))
        if tv > 0:
            te_idx.extend(range(end - tv, end))
    return tr_idx, va_idx, te_idx


def _hpo_xgb(X, y, iters: int, seed: int = 0, logger=None, idx_by_class=None, eval_frac: float | None = None):
    import numpy as np
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    rng = np.random.default_rng(seed)

    if idx_by_class and eval_frac is not None and 0.0 < eval_frac < 0.9:
        tr_idx, va_idx = _split_tail_indices(idx_by_class, eval_frac)
        Xtr, ytr = np.asarray(X)[tr_idx], np.asarray(y)[tr_idx]
        Xva, yva = np.asarray(X)[va_idx], np.asarray(y)[va_idx]
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    best_p, best_s = None, -1.0
    trials = []
    for i in range(max(1, iters)):
        params = {
            'max_depth': int(rng.integers(2, 7)),
            'n_estimators': int(rng.integers(50, 1001)),  # was 301 → allow larger ensembles
            'learning_rate': float(10 ** rng.uniform(-2.0, -0.5)),  # ~0.01..0.32
            'subsample': float(rng.uniform(0.7, 1.0)),
            'colsample_bytree': float(rng.uniform(0.7, 1.0)),
            'reg_lambda': float(10 ** rng.uniform(-3.0, 2.0)),  # was up to 10 → now up to 100
            'tree_method': 'hist',
            'n_jobs': 0,
        }
        clf = xgb.XGBClassifier(**params).fit(Xtr, ytr)
        s = float(clf.score(Xva, yva))
        trials.append((params, s))
        if logger:
            try:
                logger(i + 1, params, s)
            except Exception:
                pass
        if s > best_s:
            best_p, best_s = params, s
    return best_p, best_s, trials


def _sort_ridge_trials(trials):
    # trials: list[(alpha, score)] → ascending by score; best at bottom
    return sorted(trials, key=lambda t: t[1])


def _sort_xgb_trials(trials):
    # trials: list[(params, score)] → ascending by score; best at bottom
    return sorted(trials, key=lambda t: t[1])


def _fmt_xgb_params(p: dict) -> str:
    # Render common params with full names in a stable order
    order = [
        ("max_depth", int),
        ("n_estimators", int),
        ("learning_rate", float),
        ("subsample", float),
        ("colsample_bytree", float),
        ("reg_lambda", float),
    ]
    parts = []
    for k, typ in order:
        if k in p:
            v = p[k]
            if typ is int:
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={float(v):.3f}")
    return ", ".join(parts)


def _hpo_logreg(X, y, iters: int, seed: int = 0, logger=None, idx_by_class=None, eval_frac: float | None = None, max_iter: int = 500, solver: str = 'lbfgs'):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    if idx_by_class and eval_frac is not None and 0.0 < eval_frac < 0.9:
        tr_idx, va_idx = _split_tail_indices(idx_by_class, eval_frac)
        Xtr, ytr = np.asarray(X)[tr_idx], np.asarray(y)[tr_idx]
        Xva, yva = np.asarray(X)[va_idx], np.asarray(y)[va_idx]
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    best_C, best_s = None, -1.0
    trials = []
    for i in range(max(1, iters)):
        C = float(10 ** rng.uniform(-4, 2))
        clf = LogisticRegression(C=C, max_iter=max_iter, solver=solver).fit(Xtr, ytr)
        s = float(clf.score(Xva, yva))
        trials.append((C, s))
        if logger:
            try: logger(i+1, C, s)
            except Exception: pass
        if s > best_s:
            best_C, best_s = C, s
    return best_C, best_s, trials


def _make_sample_weights_from_labels(y, n_classes: int):
    """Balanced weights per sample: N/(C*n_c) for sample's class.

    y: array-like of class indices [0..n_classes-1]
    returns: list[float] same length as y
    """
    import numpy as _np
    y_arr = _np.asarray(y)
    if y_arr.size == 0 or n_classes <= 0:
        return []
    counts = [(y_arr == i).sum() for i in range(n_classes)]
    total = int(y_arr.size)
    # avoid div-by-zero; classes with 0 count just won't appear in y
    class_w = [float(total) / (max(1, n_classes * int(c))) for c in counts]
    return [class_w[int(ci)] for ci in y_arr]


def _train_dl(args):
    from vkb.finetune import finetune
    path = finetune(args)
    try:
        from rich.console import Console
        Console().print(f"Saved model: {path}")
    except Exception:
        print(f"Saved model: {path}")
    return path


def _prepare_classic_io(args):
    vids = list_videos(args.data)
    # Print newest video included for sanity (overall newest across labels)
    try:
        import os as _os
        latest = max(vids, key=lambda vl: _os.path.getmtime(vl[0])) if vids else None
        if latest:
            from rich.console import Console
            Console().print(f"[dim]Newest video: {_os.path.relpath(latest[0])} label={latest[1]}[/]")
    except Exception:
        pass
    labels = sorted({l for _, l in vids})
    lab2i = {l: i for i, l in enumerate(labels)}
    return vids, labels, lab2i


def _embed_videos(cons, args, vids, labels, lab2i):
    import cv2 as cv, numpy as np, os
    from time import perf_counter
    from rich.table import Table
    from rich.panel import Panel
    cons.print(Panel.fit(f"Embedding frames from {len(vids)} videos across {len(labels)} labels"))
    X, y = [], []
    feat_dim = None
    idx_by_class = {i: [] for i in range(len(labels))}
    video_slices = []  # (start, end) indices per video
    embed_params = {}
    embed = create_embedder(args.embed_model)
    cdir = _cache_dir(args.embed_model, embed_params)
    loaded_frames = 0
    computed_frames = 0
    compute_seconds = 0.0
    for i,(path, lab) in enumerate(vids, 1):
        cons.print(f"[dim]Embedding {i}/{len(vids)}: {os.path.basename(path)} ({lab})[/]")
        cache_path = _cache_path_for(cdir, args.data, path)
        cached = _cache_load(cache_path)
        if cached is not None:
            if cached.ndim == 1:
                cached = cached.reshape(1, -1)
            n = int(cached.shape[0])
            base = len(X)
            for r in range(n):
                X.append(cached[r]); y.append(lab2i[lab]); idx_by_class[lab2i[lab]].append(base + r)
            video_slices.append((base, base + n))
            loaded_frames += n
            if feat_dim is None and n:
                try:
                    feat_dim = int(cached.shape[1])
                except Exception:
                    feat_dim = None
            cons.print(f"[dim]  cache hit ({n} frames) -> {os.path.relpath(cache_path)}[/]")
            cons.print(f"[dim] ↳ done {n} frames[/]")
            continue
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            continue
        fcnt = 0
        rows = []
        t0 = perf_counter()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            f = embed(fr)
            if feat_dim is None:
                try:
                    feat_dim = len(f)
                except Exception:
                    feat_dim = None
            rows.append(f)
            fcnt += 1
        cap.release()
        if fcnt:
            arr = np.asarray(rows)
            _cache_save(cache_path, arr)
            base = len(X)
            for r in range(fcnt):
                X.append(arr[r]); y.append(lab2i[lab]); idx_by_class[lab2i[lab]].append(base + r)
            video_slices.append((base, base + fcnt))
            computed_frames += fcnt
            dt = max(perf_counter() - t0, 1e-9)
            fps = fcnt / dt
            compute_seconds += dt
            cons.print(f"[dim]  cache miss → computed {fcnt} in {dt:.2f}s ({fps:.1f} FPS) and saved {os.path.relpath(cache_path)}[/]")
        cons.print(f"[dim] ↳ done {fcnt} frames[/]")

    # Validate counts for stratified operations (eval split / HPO)
    import numpy as _np
    y_arr = _np.asarray(y)
    counts = [int((y_arr == i).sum()) for i in range(len(labels))]
    if feat_dim is not None:
        cons.print(f"[dim]Feature dim: {feat_dim}[/]")
    # Print class counts summary
    if labels:
        ct = Table(title="Class Counts")
        ct.add_column("label"); ct.add_column("frames")
        for l,c in zip(labels, counts): ct.add_row(l, str(c))
        cons.print(ct)
    if args.eval_split and 0.0 < args.eval_split < 0.9 and len(labels) > 1:
        bad = [(labels[i], c) for i, c in enumerate(counts) if c < 2]
        if bad:
            msg = "Eval split needs >=2 samples per class. Offenders: " + ", ".join(f"{l}({c})" for l,c in bad)
            raise ValueError(msg)
    # Cache summary
    from rich.table import Table as _T
    cs = _T(title="Cache Summary")
    cs.add_column("cache"); cs.add_column("loaded"); cs.add_column("computed")
    cs.add_row(cdir, str(loaded_frames), str(computed_frames))
    cons.print(cs)
    sp = _T(title="Embedding Speed")
    sp.add_column("computed"); sp.add_column("seconds"); sp.add_column("FPS")
    sp.add_row(str(computed_frames), f"{compute_seconds:.2f}", f"{(computed_frames/compute_seconds if compute_seconds>0 else 0.0):.1f}")
    cons.print(sp)
    # Expose per‑video slices to downstream splitting logic without changing signatures
    try:
        setattr(args, '_video_slices', list(video_slices))
    except Exception:
        pass
    return np.asarray(X), np.asarray(y), idx_by_class, feat_dim


def _fit_classic_and_save(cons, args, X, y, labels, idx_by_class, feat_dim):
    from rich.table import Table
    import numpy as np
    # Choose classifier + optional HPO
    if args.clf == "ridge":
        from sklearn.linear_model import RidgeClassifier
        alpha = args.alpha
        if args.hpo_alpha and len(labels) > 1 and len(X) > 10:
            bad = [(labels[i], int((np.asarray(y)==i).sum())) for i in range(len(labels)) if int((np.asarray(y)==i).sum()) < 2]
            if bad:
                raise ValueError("HPO needs >=2 samples per class. Offenders: " + ", ".join(f"{l}({c})" for l,c in bad))
            from rich.console import Console
            cons.print(f"[dim]HPO: {args.hpo_alpha} alpha trials (tail split {args.eval_split:.2f})[/]")
            frac = args.eval_split if (args.eval_split and 0.0 < args.eval_split < 0.9) else 0.2
            best_a, best_s, trials = _hpo_ridge(X, y, args.hpo_alpha, logger=lambda i,a,s: cons.print(f"[dim]  trial {i}: alpha={a:.5f} val_acc={s:.3f}[/]"), idx_by_class=idx_by_class, eval_frac=frac)
            alpha = best_a
        clf = RidgeClassifier(alpha=alpha)
        alpha_str = f"{alpha:.5f}"; xtrials = []; trials = locals().get('trials', [])
    elif args.clf == "xgb":
        import xgboost as xgb
        if args.hpo_xgb and len(labels) > 1 and len(X) > 10:
            frac = args.eval_split if (args.eval_split and 0.0 < args.eval_split < 0.9) else 0.2
            cons.print(f"[dim]HPO-XGB: {args.hpo_xgb} trials (tail split {frac:.2f})[/]")
            best_p, best_s, xtrials = _hpo_xgb(
                X,
                y,
                args.hpo_xgb,
                idx_by_class=idx_by_class,
                eval_frac=frac,
                logger=lambda i, p, s: cons.print(
                    f"[dim]  trial {i}: val_acc={s:.3f} params={_fmt_xgb_params(p)}[/]"
                ),
            )
            clf = xgb.XGBClassifier(**best_p)
        else:
            clf = xgb.XGBClassifier(max_depth=4, n_estimators=200, tree_method='hist', n_jobs=0)
            xtrials = []
        alpha_str = "-"; trials = []
    else:  # logreg
        from sklearn.linear_model import LogisticRegression
        C = args.C
        if args.hpo_logreg and len(labels) > 1 and len(X) > 10:
            frac = args.eval_split if (args.eval_split and 0.0 < args.eval_split < 0.9) else 0.2
            cons.print(f"[dim]HPO-LogReg: {args.hpo_logreg} trials (tail split {frac:.2f})[/]")
            best_C, best_s, trials = _hpo_logreg(
                X,
                y,
                args.hpo_logreg,
                idx_by_class=idx_by_class,
                eval_frac=frac,
                max_iter=args.logreg_max_iter,
                logger=lambda i, Cv, s: cons.print(f"[dim]  trial {i}: val_acc={s:.3f} C={Cv:.5f}[/]")
            )
            C = best_C
        clf = LogisticRegression(C=C, max_iter=args.logreg_max_iter, solver='lbfgs')
        alpha_str = f"C={C:.5f}"; xtrials = []
    cons.print(f"[dim]Training { _clf_display_name(args.clf) } classifier...[/]")
    # MLflow run (optional)
    import os as _os
    run = None
    mlflow = None
    if getattr(args, 'mlflow', False) and _os.getenv('VKB_MLFLOW_DISABLE','') != '1':
        try:
            import mlflow as _mlf
        except Exception as e:
            raise RuntimeError("--mlflow set but mlflow is not installed") from e
        if getattr(args, 'mlflow_uri', None):
            _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(getattr(args, 'mlflow_exp', 'vkb'))
        run_name = args.mlflow_run_name or f"{_clf_display_name(args.clf)}|{args.embed_model}"
        run = _mlf.start_run(run_name=run_name)
        mlflow = _mlf
    # Fit/evaluate and save
    val_acc = None
    X_fit, y_fit = X, y
    X_test, y_test = None, None
    _te = float(getattr(args, 'test_split', 0.0) or 0.0)
    _va = float(getattr(args, 'eval_split', 0.0) or 0.0)
    if (_te > 0.0) or (_va > 0.0):
        if args.eval_mode == "random":
            from sklearn.model_selection import train_test_split
            X_rem, Xte, y_rem, yte = (X, None, y, None)
            if _te > 0.0:
                X_rem, Xte, y_rem, yte = train_test_split(X, y, test_size=_te, stratify=y, random_state=1)
                X_test, y_test = Xte, yte
            if _va > 0.0:
                Xtr, Xva, ytr, yva = train_test_split(X_rem, y_rem, test_size=_va, stratify=y_rem, random_state=0)
                sw_tr = _make_sample_weights_from_labels(ytr, len(labels)) if str(getattr(args,'class_weights','none')) == 'auto' else None
                try:
                    clf.fit(Xtr, ytr, sample_weight=sw_tr)
                except TypeError:
                    clf.fit(Xtr, ytr)
                val_acc = float(clf.score(Xva, yva))
                X_fit, y_fit = X_rem, y_rem
        else:
            if args.eval_mode == "tail-per-video":
                slices = getattr(args, '_video_slices', None)
                if not slices:
                    raise RuntimeError("tail-per-video eval_mode requires per-video slices; ensure embedding ran")
                tr_idx, va_idx, te_idx = _split_tail_per_video_slices(slices, _va, _te)
            else:
                tr_idx, va_idx, te_idx = _split_tail_indices_three(idx_by_class, _va, _te)
            import numpy as _np
            Xarr = _np.asarray(X, dtype=object); yarr = _np.asarray(y)
            if va_idx:
                Xva = list(Xarr[va_idx]); yva = list(yarr[va_idx])
                Xtr = list(Xarr[tr_idx]); ytr = list(yarr[tr_idx])
                sw_tr = _make_sample_weights_from_labels(ytr, len(labels)) if str(getattr(args,'class_weights','none')) == 'auto' else None
                try:
                    clf.fit(Xtr, ytr, sample_weight=sw_tr)
                except TypeError:
                    clf.fit(Xtr, ytr)
                val_acc = float(clf.score(Xva, yva))
            if te_idx:
                X_test = list(Xarr[te_idx]); y_test = list(yarr[te_idx])
            X_fit, y_fit = list(Xarr[tr_idx] if tr_idx else Xarr), list(yarr[tr_idx] if tr_idx else yarr)
    sw_fit = _make_sample_weights_from_labels(y_fit, len(labels)) if str(getattr(args,'class_weights','none')) == 'auto' else None
    try:
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)
    except TypeError:
        clf.fit(X_fit, y_fit)
    clf_name = _clf_display_name(args.clf)
    bundle = {"clf": clf, "labels": labels, "clf_name": clf_name, "embed_model": args.embed_model, "embed_params": {}}
    base_models = (getattr(args, 'models_dir', None) or "models")
    name_parts = [clf_name, args.embed_model]
    if val_acc is not None:
        name_parts.append(f"val{float(val_acc):.3f}")
    path = save_model(bundle, name_parts, base_dir=base_models)
    # Optional test accuracy
    test_acc = None
    if X_test is not None and len(X_test) > 0:
        try:
            test_acc = float(clf.score(X_test, y_test))
        except Exception:
            test_acc = None
    side = {
        "clf_name": clf_name,
        "embed_model": args.embed_model,
        "eval_split": getattr(args, 'eval_split', None),
        "eval_mode": getattr(args, 'eval_mode', None),
        "labels": labels,
        "feat_dim": int(feat_dim) if feat_dim is not None else None,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "frames": len(y),
        "hparams": {"alpha": getattr(args,'alpha',None) if args.clf=="ridge" else None, "C": getattr(args,'C',None) if args.clf=="logreg" else None},
    }
    meta_path = save_sidecar(path, side)
    if mlflow is not None:
        p = {'clf': clf_name, 'embed_model': args.embed_model, 'feat_dim': int(feat_dim) if feat_dim is not None else -1, 'frames': int(len(y)), 'classes': int(len(labels)), 'eval_split': float(args.eval_split), 'eval_mode': str(args.eval_mode)}
        if args.clf == 'ridge': p['alpha'] = float(alpha)
        if args.clf == 'logreg': p['C'] = float(C)
        try: mlflow.log_params(p)
        except Exception: pass
        if val_acc is not None:
            try: mlflow.log_metric('val_acc', float(val_acc), step=0)
            except Exception: pass
        try:
            mlflow.log_artifact(path)
            mlflow.log_artifact(meta_path)
        except Exception:
            pass
        try: mlflow.end_run()
        except Exception: pass
    t = Table(title="Training Summary")
    t.add_column("classifier"); t.add_column("embedder"); t.add_column("alpha"); t.add_column("feat_dim"); t.add_column("frames"); t.add_column("classes"); t.add_column("val_acc"); t.add_column("test_acc"); t.add_column("saved")
    t.add_row(clf_name, args.embed_model, alpha_str, str(feat_dim if feat_dim is not None else "-"), str(len(y)), str(len(labels)), f"{val_acc:.3f}" if val_acc is not None else "-", f"{test_acc:.3f}" if test_acc is not None else "-", path)
    cons.print(t)
    try: cons.print(f"Saved model: {path}")
    except Exception: pass
    if args.clf == "ridge" and locals().get('trials'):
        ht = Table(title="HPO (alpha)")
        ht.add_column("trial"); ht.add_column("alpha"); ht.add_column("val_acc")
        for i, (a, s) in enumerate(_sort_ridge_trials(trials)):
            ht.add_row(str(i+1), f"{a:.5f}", f"{s:.3f}")
        cons.print(ht)
    if args.clf == "xgb" and xtrials:
        ht = Table(title="HPO-XGB")
        ht.add_column("trial"); ht.add_column("val_acc"); ht.add_column("params")
        for i, (p, s) in enumerate(_sort_xgb_trials(xtrials)):
            ht.add_row(str(i+1), f"{s:.3f}", _fmt_xgb_params(p))
        cons.print(ht)
    if args.clf == "logreg" and locals().get('trials'):
        ht = Table(title="HPO-LogReg (C)")
        ht.add_column("trial"); ht.add_column("C"); ht.add_column("val_acc")
        for i, (Cv, s) in enumerate(_sort_ridge_trials(trials)):
            ht.add_row(str(i+1), f"{Cv:.5f}", f"{s:.3f}")
        cons.print(ht)
    return path


def _train_classic(args):
    trainer = ClassicTrainer(args)
    return trainer.run()


class ClassicTrainer:
    def __init__(self, args, console=None):
        self.args = args
        if console is None:
            from rich.console import Console
            console = Console()
        self.cons = console
        # Lazily populated
        self.vids = self.labels = self.lab2i = None
        self.X = self.y = self.idx_by_class = None
        self.feat_dim = None

    def prepare(self):
        self.vids, self.labels, self.lab2i = _prepare_classic_io(self.args)
        return self

    def embed(self):
        X, y, idx_by_class, feat_dim = _embed_videos(self.cons, self.args, self.vids, self.labels, self.lab2i)
        self.X, self.y, self.idx_by_class, self.feat_dim = X, y, idx_by_class, feat_dim
        return self

    def fit_and_save(self):
        return _fit_classic_and_save(self.cons, self.args, self.X, self.y, self.labels, self.idx_by_class, self.feat_dim)

    def run(self):
        return self.prepare().embed().fit_and_save()


def _train_mediapipe_logreg(args):
    """Train Logistic Regression on MediaPipe hand landmark distance features.

    Feature per sample: upper‑triangle pairwise distances between 21 landmarks,
    L1‑normalized each frame. Keeps code minimal; no hidden fallbacks.
    """
    from rich.console import Console
    cons = Console()
    from vkb.io import list_videos
    vids = list_videos(args.data)
    if not vids:
        raise RuntimeError(f"no videos under {args.data}")
    labels = sorted({lab for _, lab in vids})
    lab2i = {lab: i for i, lab in enumerate(labels)}
    # Gather features per video
    from vkb.landmarks import extract_features_for_video
    import numpy as np
    X, y = [], []
    idx_by_class = {i: [] for i in range(len(labels))}
    video_slices = []  # for tail-per-video split
    start = 0
    for path, lab in vids:
        feats = extract_features_for_video(path, stride=int(args.mp_stride), max_frames=int(args.mp_max_frames))
        if feats.size == 0:
            continue
        n = feats.shape[0]
        ci = lab2i[lab]
        X.extend(feats)
        y.extend([ci] * n)
        idx_by_class[ci].extend(range(start, start + n))
        video_slices.append((start, start + n))
        start += n
    if not X:
        raise RuntimeError("no landmark features extracted; check mediapipe installation and video content")
    cons.print(f"[dim]Prepared features: frames={len(y)} classes={len(labels)} feat_dim={len(X[0])}[/]")
    # Choose C via optional HPO
    from sklearn.linear_model import LogisticRegression
    C = float(args.C)
    val_acc = None
    trials = []
    if int(getattr(args, 'hpo_logreg', 0) or 0) > 0:
        C, val_acc, trials = _hpo_logreg(X, y, iters=int(args.hpo_logreg), idx_by_class=idx_by_class, eval_frac=float(args.eval_split), max_iter=int(args.logreg_max_iter))
    clf = LogisticRegression(C=float(C), max_iter=int(args.logreg_max_iter), solver='lbfgs')
    # Split train/val/test
    X_fit, y_fit = X, y
    X_test, y_test = None, None
    _te = float(getattr(args, 'test_split', 0.0) or 0.0)
    _va = float(getattr(args, 'eval_split', 0.0) or 0.0)
    if (_te > 0.0) or (_va > 0.0):
        import numpy as _np
        if args.eval_mode == "tail-per-video":
            tr_idx, va_idx, te_idx = _split_tail_per_video_slices(video_slices, _va, _te)
        else:
            tr_idx, va_idx, te_idx = _split_tail_indices_three(idx_by_class, _va, _te)
        Xarr = _np.asarray(X, dtype=object); yarr = _np.asarray(y)
        if va_idx:
            Xva = list(Xarr[va_idx]); yva = list(yarr[va_idx])
            Xtr = list(Xarr[tr_idx]); ytr = list(yarr[tr_idx])
            clf.fit(Xtr, ytr)
            val_acc = float(clf.score(Xva, yva))
        if te_idx:
            X_test = list(Xarr[te_idx]); y_test = list(yarr[te_idx])
        X_fit, y_fit = list(Xarr[tr_idx] if tr_idx else Xarr), list(yarr[tr_idx] if tr_idx else yarr)
    cons.print(f"[dim]Training mp_logreg (C={C:.5f})...[/]")
    clf.fit(X_fit, y_fit)
    from vkb.artifacts import save_model, save_sidecar
    name_parts = ["mp_logreg", "mediapipe_hand"]
    if val_acc is not None:
        name_parts.append(f"val{float(val_acc):.3f}")
    path = save_model({"clf": clf, "labels": labels, "clf_name": "mp_logreg", "embed_model": "mediapipe_hand", "embed_params": {}}, name_parts)
    test_acc = None
    if X_test is not None:
        try:
            test_acc = float(clf.score(X_test, y_test))
        except Exception:
            test_acc = None
    save_sidecar(path, {
        "clf_name": "mp_logreg",
        "embed_model": "mediapipe_hand",
        "eval_split": getattr(args, 'eval_split', None),
        "eval_mode": getattr(args, 'eval_mode', None),
        "labels": labels,
        "feat_dim": int(len(X[0])),
        "val_acc": val_acc,
        "test_acc": test_acc,
        "frames": len(y),
        "hparams": {"C": float(C), "mp_stride": int(getattr(args,'mp_stride',5)), "mp_max_frames": int(getattr(args,'mp_max_frames',200)), "feat_norm": "l1", "pairs": "upper_xy", "landmarks": 21},
    })
    # Print tiny HPO table if we ran it
    if trials:
        from rich.table import Table
        ht = Table(title="HPO-LogReg (C)")
        ht.add_column("trial"); ht.add_column("C"); ht.add_column("val_acc")
        for i, (Cv, s) in enumerate(_sort_ridge_trials(trials)):
            ht.add_row(str(i+1), f"{Cv:.5f}", f"{s:.3f}")
        cons.print(ht)
    return path


def train(args):
    # Build unified config and attach for downstream consumers
    try:
        from vkb.config import make_config
        args.cfg = make_config(args)
    except Exception:
        pass
    # Propagate diag flag to DL finetune path
    try:
        if getattr(args, 'diag_system', False):
            setattr(args, 'diag_system', True)
    except Exception:
        pass
    if args.clf == "dl":
        return _train_dl(args)
    if args.clf == "mp_logreg":
        return _train_mediapipe_logreg(args)
    return _train_classic(args)


# --- tiny helpers (kept here for tests) ---
def _apply_thread_caps(args):
    """If --cap-threads is set, reduce internal thread pools to 1.
    Sets common env vars and tries cv2/torch setters when present.
    """
    if not getattr(args, 'cap_threads', False):
        return
    import os
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    try:
        import cv2 as _cv
        try: _cv.setNumThreads(1)
        except Exception: pass
    except Exception:
        pass
    try:
        import torch as _torch
        try:
            _torch.set_num_threads(1)
        except Exception:
            pass
        try:
            _torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass

def _maybe_nice(args):
    """If --nice > 0, lower priority a bit (Linux). No-op elsewhere."""
    try:
        n = int(getattr(args, 'nice', 0) or 0)
        if n > 0:
            import os
            try:
                os.nice(n)
            except Exception:
                pass
    except Exception:
        pass


def main():
    args = parse_args()
    _apply_thread_caps(args)
    _maybe_nice(args)
    # Guardrail: avoid accidental training on test labels ['A','B'] when running real CLI with default models dir
    try:
        vids = list_videos(args.data)
        labels = sorted({l for _, l in vids})
        if labels == ['A', 'B'] and not args.allow_test_labels:
            from rich.console import Console
            cons = Console()
            cons.print("[bold red]Refusing to train on labels ['A','B'] without --allow-test-labels[/]")
            raise SystemExit(2)
    except Exception:
        pass
    train(args)


if __name__ == "__main__":
    main()
