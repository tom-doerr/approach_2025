#!/usr/bin/env python3
"""
Minimal HPO for MediaPipe‑XGBoost (mp_xgb):
- Maximizes validation accuracy and reports inference throughput.
- Keeps code tiny: random search, tail/tail‑per‑video split, no fallbacks.

Usage example:
  python hpo_mp_xgb.py --data data --eval-split 0.2 --trials 20 --mp-stride 1
"""
from __future__ import annotations
import argparse, time, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


def _list_videos(root: str):
    from vkb.io import list_videos
    return list_videos(root)


def _extract_features(path: str, stride: int, max_frames: int, use_stub: bool=False):
    if use_stub:
        # Tiny, deterministic synthetic features: one-hot by filename prefix
        import numpy as np, os
        v = np.zeros(210, dtype=float)
        v[0 if os.path.basename(path).lower().startswith('a') else 1] = 1.0
        k = max(1, int(max_frames) or 6)
        return np.stack([v.copy() for _ in range(k)], axis=0)
    from vkb.landmarks import extract_features_for_video
    return extract_features_for_video(path, stride=stride, max_frames=max_frames)


def _split_tail_indices(idx_by_class: Dict[int, List[int]], frac: float) -> Tuple[List[int], List[int]]:
    tr, va = [], []
    for _, idxs in idx_by_class.items():
        n = len(idxs)
        if n <= 1:
            tr.extend(idxs); continue
        v = max(1, int(round(n * frac)))
        v = min(n - 1, v)
        tr.extend(idxs[:-v]); va.extend(idxs[-v:])
    return tr, va


def _split_tail_per_video_slices(video_slices: List[Tuple[int,int]], val_frac: float):
    vids = list(video_slices)
    lens = [max(0, e - s) for (s, e) in vids]
    T = sum(lens)
    if T == 0:
        return [], []
    # Hamilton apportionment
    import math
    targ = [l * float(val_frac) for l in lens]
    base = [int(math.floor(t)) for t in targ]
    need = int(round(T * float(val_frac))) - sum(base)
    rem = sorted([(i, targ[i] - base[i]) for i in range(len(lens))], key=lambda x: x[1], reverse=True)
    for i, _ in rem:
        if need <= 0: break
        base[i] += 1; need -= 1
    tr, va = [], []
    for (s, e), L, v in zip(vids, lens, base):
        h = max(0, L - v)
        tr.extend(range(s, s + h))
        if v: va.extend(range(e - v, e))
    return tr, va


@dataclass
class Trial:
    params: Dict[str, Any]
    val_acc: float
    train_ms: float
    infer_ms: float
    infer_samples: int

    @property
    def infer_sps(self) -> float:
        return (self.infer_samples / self.infer_ms * 1000.0) if self.infer_ms > 0 else 0.0


def _sample_params(rng) -> Dict[str, Any]:
    import numpy as np
    return {
        'max_depth': int(rng.integers(2, 4)),  # 2..3 inclusive
        'n_estimators': int(rng.integers(400, 4001)),
        'learning_rate': float(10 ** rng.uniform(-2.3, -0.5)),  # ~0.005..0.3
        'subsample': float(rng.uniform(0.5, 1.0)),
        'colsample_bytree': float(rng.uniform(0.5, 1.0)),
        'reg_lambda': float(10 ** rng.uniform(-4.0, 2.0)),
        'tree_method': 'hist',
        'n_jobs': 0,
    }


def run_hpo(data: str, eval_split: float, trials: int, seed: int, eval_mode: str,
            mp_stride: int, mp_max_frames: int, use_stub: bool=False,
            min_sps: float = 0.0, early_rounds: int = 50) -> Dict[str, Any]:
    import numpy as np
    import xgboost as xgb

    vids = _list_videos(data)
    if not vids:
        raise RuntimeError(f"no videos under {data}")
    labels = sorted({lab for _, lab in vids})
    lab2i = {lab: i for i, lab in enumerate(labels)}

    X, y = [], []
    idx_by_class = {i: [] for i in range(len(labels))}
    video_slices = []
    start = 0
    for path, lab in vids:
        feats = _extract_features(path, stride=int(mp_stride), max_frames=int(mp_max_frames), use_stub=use_stub)
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

    X = np.asarray(X); y = np.asarray(y)
    if eval_mode == 'tail-per-video':
        tr_idx, va_idx = _split_tail_per_video_slices(video_slices, float(eval_split))
    else:
        tr_idx, va_idx = _split_tail_indices(idx_by_class, float(eval_split))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    rng = np.random.default_rng(seed)
    trials_out: List[Trial] = []
    for i in range(max(1, int(trials))):
        p = _sample_params(rng)
        t0 = time.perf_counter();
        clf = xgb.XGBClassifier(**p)
        fit_ok = False
        # Try common early-stopping signatures; fall back to plain fit if unsupported
        try:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=int(early_rounds))
            fit_ok = True
        except TypeError:
            try:
                from xgboost.callback import EarlyStopping
                clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                        callbacks=[EarlyStopping(rounds=int(early_rounds), save_best=True)])
                fit_ok = True
            except Exception:
                pass
        if not fit_ok:
            clf.fit(Xtr, ytr)
        train_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter();
        probs = clf.predict_proba(Xva)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        pred = probs.argmax(axis=1)
        acc = float((pred == yva).mean())
        trials_out.append(Trial(p, acc, train_ms, infer_ms, int(len(Xva))))

    # Pareto front (maximize acc & sps)
    pareto: List[int] = []
    for i, a in enumerate(trials_out):
        dom = False
        for j, b in enumerate(trials_out):
            if j == i: continue
            if (b.val_acc >= a.val_acc and b.infer_sps >= a.infer_sps) and (
                b.val_acc > a.val_acc or b.infer_sps > a.infer_sps):
                dom = True; break
        if not dom: pareto.append(i)

    # Simple correlations (directional hints)
    def _corr(x, y):
        # Spearman rank corr without SciPy
        xr = np.argsort(np.argsort(x))
        yr = np.argsort(np.argsort(y))
        xc = xr - xr.mean(); yc = yr - yr.mean()
        num = float((xc * yc).sum())
        den = float((xc**2).sum()**0.5 * (yc**2).sum()**0.5)
        return (num / den) if den > 0 else 0.0

    keys = ['max_depth','n_estimators','learning_rate','subsample','colsample_bytree','reg_lambda']
    acc_corr = {}; sps_corr = {}
    for k in keys:
        xs = [t.params[k] for t in trials_out]
        acc_corr[k] = _corr(xs, [t.val_acc for t in trials_out])
        sps_corr[k] = _corr(xs, [t.infer_sps for t in trials_out])

    # Best indices
    best_acc_idx = max(range(len(trials_out)), key=lambda i: trials_out[i].val_acc)
    ge = [i for i,t in enumerate(trials_out) if t.infer_sps >= float(min_sps)]
    best_acc_ge_min_idx = (max(ge, key=lambda i: trials_out[i].val_acc) if ge else None)

    return {
        'labels': labels,
        'n_samples': int(len(X)),
        'eval_split': float(eval_split),
        'eval_mode': str(eval_mode),
        'trials': [
            {
                'params': t.params,
                'val_acc': round(t.val_acc, 6),
                'train_ms': round(t.train_ms, 2),
                'infer_ms': round(t.infer_ms, 2),
                'infer_sps': round(t.infer_sps, 2),
            } for t in trials_out
        ],
        'pareto_idx': pareto,
        'best_acc_idx': best_acc_idx,
        'best_acc_ge_min_sps_idx': best_acc_ge_min_idx,
        'acc_corr': acc_corr,
        'sps_corr': sps_corr,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--eval-split', type=float, default=0.2)
    p.add_argument('--eval-mode', choices=['tail','tail-per-video'], default='tail-per-video')
    p.add_argument('--trials', type=int, default=15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mp-stride', type=int, default=1)
    p.add_argument('--mp-max-frames', type=int, default=0)
    p.add_argument('--min-sps', type=float, default=0.0, help='Prefer/print configs with >= this throughput (samples/s)')
    p.add_argument('--early-rounds', type=int, default=50)
    p.add_argument('--use-stub', action='store_true', help='Use synthetic landmark features (for quick local study)')
    args = p.parse_args(argv)
    res = run_hpo(args.data, args.eval_split, args.trials, args.seed, args.eval_mode, args.mp_stride, args.mp_max_frames,
                  use_stub=args.use_stub, min_sps=args.min_sps, early_rounds=args.early_rounds)
    # Pretty summary
    print(f"Trials: {len(res['trials'])}, samples={res['n_samples']}, eval_split={res['eval_split']:.2f} ({res['eval_mode']})")
    # Prefer configs meeting min-sps; fall back to overall
    filtered = [t for t in res['trials'] if t['infer_sps'] >= args.min_sps]
    pool = filtered if filtered else res['trials']
    top = sorted(pool, key=lambda t: (t['val_acc'], t['infer_sps']))[-5:]
    for i, t in enumerate(top, 1):
        p = t['params']
        print(f"top{i}: acc={t['val_acc']:.3f} sps={t['infer_sps']:.0f} depth={p['max_depth']} est={p['n_estimators']} lr={p['learning_rate']:.3f} sub={p['subsample']:.2f} col={p['colsample_bytree']:.2f} lam={p['reg_lambda']:.3f}")
    print("RESULTS_JSON:")
    print(json.dumps(res))


if __name__ == '__main__':
    main()
