#!/usr/bin/env python3
import argparse, pickle
from vkb.artifacts import latest_model
import os
from vkb.emb import create_embedder


def _make_preprocess(insz: int, mean, std, dev: str):
    import numpy as np
    import cv2 as cv
    import torch
    mean = np.array(mean, dtype="float32")[:, None, None]
    std = np.array(std, dtype="float32")[:, None, None]
    def preprocess(frame):
        fr = cv.cvtColor(cv.resize(frame, (insz, insz)), cv.COLOR_BGR2RGB)
        x = (fr.astype("float32")/255.0).transpose(2,0,1)
        x = (x - mean) / std
        return torch.from_numpy(x).unsqueeze(0).to(dev)
    return preprocess


def _apply_mjpg_mode(cap, cam_index: int = 0):
    # Mirror record_video: prefer first MJPG mode via v4l2-ctl on Linux
    try:
        import sys as _sys, cv2 as cv
        if str(getattr(_sys, 'platform', '')) != 'linux':
            return
        try:
            from record_video import _pick_first_mjpg_mode
            choice = _pick_first_mjpg_mode(int(cam_index))
        except Exception:
            choice = None
        if hasattr(cv, 'CAP_PROP_FOURCC'):
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        if choice is not None:
            ww, hh, ff = choice
            cap.set(getattr(cv, 'CAP_PROP_FRAME_WIDTH', 3), int(ww))
            cap.set(getattr(cv, 'CAP_PROP_FRAME_HEIGHT', 4), int(hh))
            cap.set(getattr(cv, 'CAP_PROP_FPS', 5), int(ff))
        else:
            # Match recorder's simple fallback
            cap.set(getattr(cv, 'CAP_PROP_FRAME_WIDTH', 3), 1920)
            cap.set(getattr(cv, 'CAP_PROP_FRAME_HEIGHT', 4), 1080)
            cap.set(getattr(cv, 'CAP_PROP_FPS', 5), 30)
    except Exception:
        return


def parse_args():
    p = argparse.ArgumentParser(description="Live inference: show camera and predicted label")
    p.add_argument("--model-path", default=None, help="path to saved .pkl; defaults to latest in models/")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto", help="DL: device (for finetune models)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    p.add_argument("--unsafe-load", action="store_true", help="allow torch.load(weights_only=False)/pickle for trusted files if safe load fails")
    return p.parse_args()

def load_bundle(path: str, unsafe: bool = False):
    # Prefer classic pickle first (sklearn bundles), then safe torch.load for DL
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e_pickle:
        # Finetune checkpoints (DL): use safe weights_only load first
        try:
            import torch
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e_safe:
            if not unsafe:
                raise RuntimeError(f"safe load failed: {e_safe}. Re-run with --unsafe-load if you trust this file.")
            # Explicit opt-in: allow full unpickling for trusted artifacts
            try:
                import torch
                return torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)


def choose_model_path(cli_path: str | None, base_dir: str = "models") -> str:
    base = os.getenv('VKB_MODELS_DIR', base_dir)
    return cli_path or latest_model(base)


def _summarize_classic(clf):
    try:
        name = type(clf).__name__
        # Known sklearn/XGB keys (keep concise)
        if name == 'LogisticRegression':
            return f"type={name} C={getattr(clf,'C',None)} solver={getattr(clf,'solver',None)} max_iter={getattr(clf,'max_iter',None)}"
        if name == 'RidgeClassifier':
            return f"type={name} alpha={getattr(clf,'alpha',None)}"
        if name == 'XGBClassifier':
            gp = getattr(clf, 'get_xgb_params', None)
            p = gp() if gp else getattr(clf, 'get_params', lambda: {})()
            def g(k, default=None):
                return p.get(k, default)
            return (
                f"type={name} est={g('n_estimators', '?')} depth={g('max_depth','?')} lr={g('learning_rate','?')} "
                f"sub={g('subsample','?')} col={g('colsample_bytree','?')} reg_lambda={g('reg_lambda','?')}"
            )
        # Generic sklearn: show class and a couple of params if available
        gp = getattr(clf, 'get_params', None)
        if gp:
            p = gp()
            some = []
            for k in ('alpha','C','max_iter','n_estimators','max_depth'):
                if k in p: some.append(f"{k}={p[k]}")
            return "type=" + name + (" " + " ".join(some) if some else '')
        return f"type={name}"
    except Exception:
        return "type=unknown"


def _summarize_dl(bundle):
    try:
        model_name = bundle.get('model_name') or bundle.get('backbone') or '?'
        insz = bundle.get('input_size') or '?'
        norm = bundle.get('normalize') or {}
        drop = bundle.get('dropout', None)
        dropp = bundle.get('drop_path', None)
        extra = []
        if drop is not None: extra.append(f"dropout={drop}")
        if dropp is not None: extra.append(f"drop_path={dropp}")
        return f"backbone={model_name} input={insz} " + (" ".join(extra))
    except Exception:
        return "backbone=?"


def _check_feat_dim(clf, feat_len: int):
    n = getattr(clf, "n_features_in_", None)
    if n is not None and n != feat_len:
        raise ValueError(f"feature dim mismatch: classifier expects {n}, got {feat_len}. Pick the correct model with --model-path or retrain.")


def _format_timings(fps: float, proc_ms: float, emb_ms: float, clf_ms: float) -> str:
    return f"FPS: {fps:.1f}  Proc: {proc_ms:.1f} ms  Emb: {emb_ms:.1f} ms  Clf: {clf_ms:.1f} ms"


def _xgb_fast_predict(clf, feat):
    try:
        import numpy as np
        booster = clf.get_booster()
        out = booster.inplace_predict(np.asarray(feat, dtype=np.float32).reshape(1, -1))
        arr = np.asarray(out)
        if arr.ndim == 0:
            return int(arr)
        if arr.ndim == 1:
            if arr.size == 1:
                # binary probability for class 1
                return 1 if float(arr[0]) >= 0.5 else 0
            # multiclass probabilities
            return int(arr.argmax())
        # multi: probabilities per class
        return int(arr.reshape(-1).argmax())
    except Exception:
        return None


def _xgb_probs(clf, feat):
    try:
        import numpy as np
        # Prefer sklearn API when available
        if hasattr(clf, 'predict_proba'):
            p = clf.predict_proba([feat])[0]
            return np.asarray(p, dtype=float)
        # Fallback to booster inplace_predict probabilities
        booster = getattr(clf, 'get_booster', lambda: None)()
        if booster is None:
            return None
        out = booster.inplace_predict(np.asarray(feat, dtype=np.float32).reshape(1, -1))
        arr = np.asarray(out)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            if arr.size == 1:
                p1 = float(arr[0])
                return np.array([1.0 - p1, p1], dtype=float)
            return arr.astype(float)
        return arr.reshape(-1).astype(float)
    except Exception:
        return None


def _scores_probs_sklearn(clf, feat):
    import numpy as np
    raw = None
    prob = None
    if hasattr(clf, 'decision_function'):
        try:
            raw = clf.decision_function([feat])
            raw = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            raw = None
    if hasattr(clf, 'predict_proba'):
        try:
            prob = clf.predict_proba([feat])[0]
            prob = np.asarray(prob, dtype=float).reshape(-1)
        except Exception:
            prob = None
    # Special case XGB to ensure probabilities exist
    if prob is None and ('xgboost' in str(type(clf)).lower()):
        prob = _xgb_probs(clf, feat)
    return raw, prob


def _make_mp_embedder():
    """Return (embed_fn, cleanup) for MediaPipe landmark features.

    embed_fn(frame) -> (feature (210,) | None, pts_px | None)
    where pts_px is a list of (x,y) pixel coords for drawing.
    """
    import numpy as np
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for mp_logreg inference") from e
    from vkb.landmarks import pairwise_distance_features
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    def embed(frame):
        rgb = frame[:, :, ::-1]
        res = hands.process(rgb)
        if not getattr(res, 'multi_hand_landmarks', None):
            return None, None
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y, getattr(p, 'z', 0.0)] for p in lm], dtype=float)
        # pixel coords for overlay
        h, w = frame.shape[:2]
        pts_px = [(int(min(max(px * w, 0), w - 1)), int(min(max(py * h, 0), h - 1))) for px, py, _ in pts]
        return pairwise_distance_features(pts), pts_px
    def cleanup():
        try: hands.close()
        except Exception: pass
    return embed, cleanup


def main():
    import cv2 as cv
    from time import perf_counter
    from rich.console import Console
    args = parse_args()
    path = choose_model_path(args.model_path, "models")
    try:
        bundle = load_bundle(path, unsafe=bool(getattr(args, 'unsafe_load', False)))
    except Exception as e:
        from rich.console import Console
        Console().print(f"[red]Failed to load model {path}: {e}[/]")
        return
    labels = bundle["labels"]
    clf_name = str(bundle.get('clf_name', ''))
    title = f"infer: {clf_name or 'dl'}"

    # Two paths: classic (sklearn) vs finetuned DL (state_dict)
    clf = bundle.get("clf", None)
    dl_model = None
    preprocess = None
    is_dl = (clf_name == 'finetune') or ((clf is None) and ("state_dict" in bundle) and (bundle.get("state_dict") is not None))
    if is_dl:
        # Finetuned path
        try:
            import torch, timm
            dev = args.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Install CUDA PyTorch or use --device cpu.")
            model_name = bundle.get("model_name", "mobilenetv3_small_100")
            dl_model = timm.create_model(model_name, pretrained=False, num_classes=len(labels))
            dl_model.load_state_dict(bundle["state_dict"], strict=False)
            dl_model.eval().to(dev)
            insz = int(bundle.get("input_size", 224))
            norm = bundle.get("normalize", {})
            mean = norm.get("mean", [0.485, 0.456, 0.406])
            std = norm.get("std", [0.229, 0.224, 0.225])
            preprocess = _make_preprocess(insz, mean, std, dev)
            title = f"infer: finetune | {model_name}"
        except RuntimeError as e:
            if str(e).startswith("CUDA requested"):
                raise
            Console().print(f"[red]DL infer disabled: RuntimeError: {e}[/]")
            return
        except Exception as e:
            Console().print(f"[red]DL infer disabled: {type(e).__name__}: {e}[/]")
            return
    else:
        # Classic path (incl. mp_logreg)
        model_name = bundle.get("embed_model", "?")
        title = f"infer: {clf_name} | {model_name}"
        mp_mode = (clf_name == 'mp_logreg') or (model_name == 'mediapipe_hand')
        mp_cleanup = None
        if mp_mode:
            embed, mp_cleanup = _make_mp_embedder()
        else:
            embed = create_embedder(model_name)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera not available")
    _apply_mjpg_mode(cap, 0)
    cons = Console()
    cons.print(f"[bold green]Loaded[/]: [magenta]{path}[/]")
    if clf is not None:
        cons.print(f"[bold]Model[/]: [cyan]{bundle['clf_name']}[/] | [cyan]{model_name}[/]")
        cons.print(f"[dim]Params: {_summarize_classic(clf)}[/]")
    else:
        cons.print(f"[bold]Model[/]: [cyan]finetune[/] | [cyan]{bundle.get('model_name','?')}[/]")
        cons.print(f"[dim]Params: {_summarize_dl(bundle)}[/]")
    fps, last_t = 0.0, perf_counter()
    emb_ms = 0.0  # preprocess or embed time
    proc_ms = 0.0
    clf_ms = 0.0  # classifier or model time
    emb_label = 'Emb'; clf_label = 'Clf'
    frames_left = int(args.frames)
    while True:
        frame_t0 = perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        # Reset per-frame diagnostics to avoid stale overlays
        raw = None; prob = None; lm_pts = None; lab = None
        if clf is not None:
            emb_t0 = perf_counter()
            if 'mp_mode' in locals() and mp_mode:
                feat, lm_pts = embed(frame)
            else:
                feat = embed(frame)
            emb_ms = (perf_counter() - emb_t0) * 1000.0
            if feat is None:
                lab = 'no_hand'
                pred_idx = 0
                clf_ms = 0.0
            else:
                try:
                    _check_feat_dim(clf, len(feat))
                except ValueError as e:
                    cons.print(f"[bold red]{e}[/]")
                    break
                clf_t0 = perf_counter()
                # Compute prediction and gather raw/prob vectors when available
                raw, prob = _scores_probs_sklearn(clf, feat)
                if prob is not None:
                    import numpy as _np
                    pred_idx = int(_np.argmax(prob))
                else:
                    pi = _xgb_fast_predict(clf, feat)
                    if pi is None:
                        pred = clf.predict([feat])[0]
                        pred_idx = int(pred) if not hasattr(pred, 'item') else int(pred.item())
                    else:
                        pred_idx = int(pi)
                clf_ms = (perf_counter() - clf_t0) * 1000.0
                lab = labels[pred_idx]
        else:
            # DL path
            emb_t0 = perf_counter()
            x = preprocess(frame)
            emb_ms = (perf_counter() - emb_t0) * 1000.0
            import torch
            with torch.no_grad():
                clf_t0 = perf_counter()
                logits = dl_model(x)
                pred_idx = int(logits.argmax(dim=1).item())
                clf_ms = (perf_counter() - clf_t0) * 1000.0
            lab = labels[pred_idx]
        # update fps
        now = perf_counter(); dt = now - last_t; last_t = now
        proc_ms = (now - frame_t0) * 1000.0
        if dt > 0:
            inst = 1.0/dt
            fps = inst if fps == 0.0 else (0.9*fps + 0.1*inst)
        # overlay
        # Decide labels for timing overlay (make MP/XGB explicit when applicable)
        try:
            if clf is not None:
                if 'mp_mode' in locals() and mp_mode:
                    emb_label = 'MP'
                else:
                    emb_label = 'Emb'
                tname = str(type(clf)).lower()
                if 'xgb' in tname or 'xgboost' in tname:
                    clf_label = 'XGB'
                elif 'logistic' in tname:
                    clf_label = 'LogReg'
                else:
                    clf_label = 'Clf'
            else:
                emb_label = 'Pre'; clf_label = 'DL'
        except Exception:
            emb_label = 'Emb'; clf_label = 'Clf'

        if 'lm_pts' in locals() and lm_pts:
            try:
                for (x, y) in lm_pts:
                    cv.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1, lineType=cv.LINE_AA)
            except Exception:
                pass
        cv.putText(frame, str(lab), (12, 28), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"FPS: {fps:.1f}", (12, 56), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Proc: {proc_ms:.1f} ms  {emb_label}: {emb_ms:.1f} ms  {clf_label}: {clf_ms:.1f} ms", (12, 82), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv.LINE_AA)
        # Show per-class raw and probability vectors when available
        try:
            y0 = 108
            if clf is not None:
                # raw/prob from sklearn path
                if 'raw' in locals() and raw is not None:
                    txt = "raw=[" + " ".join(f"{float(v):.2f}" for v in raw.tolist()) + "]"
                    cv.putText(frame, txt, (12, y0), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 2, cv.LINE_AA); y0 += 22
                if 'prob' in locals() and prob is not None:
                    txt = "prob=[" + " ".join(f"{float(p):.2f}" for p in prob.tolist()) + "]"
                    cv.putText(frame, txt, (12, y0), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv.LINE_AA); y0 += 22
                    # Also print class-named probabilities
                    try:
                        named = "prob_names=" + " ".join(f"{labels[i]}={float(prob[i]):.2f}" for i in range(len(labels)))
                        cv.putText(frame, named, (12, y0), cv.FONT_HERSHEY_SIMPLEX, 0.6, (160, 255, 160), 2, cv.LINE_AA); y0 += 22
                    except Exception:
                        pass
            else:
                # DL path: show logits and softmax
                import torch
                p = torch.softmax(logits, dim=1).squeeze(0).tolist()
                r = logits.squeeze(0).tolist()
                txt = "raw=[" + " ".join(f"{float(v):.2f}" for v in r) + "]"
                cv.putText(frame, txt, (12, 108), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 2, cv.LINE_AA)
                txt = "prob=[" + " ".join(f"{float(q):.2f}" for q in p) + "]"
                cv.putText(frame, txt, (12, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv.LINE_AA)
                try:
                    named = "prob_names=" + " ".join(f"{labels[i]}={float(p[i]):.2f}" for i in range(len(labels)))
                    cv.putText(frame, named, (12, 152), cv.FONT_HERSHEY_SIMPLEX, 0.6, (160, 255, 160), 2, cv.LINE_AA)
                except Exception:
                    pass
        except Exception:
            pass
        cv.imshow(title, frame)
        k = cv.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if frames_left > 0:
            frames_left -= 1
            if frames_left <= 0:
                break
    cap.release(); cv.destroyAllWindows()
    try:
        if locals().get('mp_cleanup'): mp_cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    main()
