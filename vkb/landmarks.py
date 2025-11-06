import os
import numpy as np
import json


def _fingerprint(video_path: str):
    try:
        st = os.stat(video_path)
        return int(st.st_size), int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)))
    except Exception:
        return None, None


def _cache_path(video_path: str, stride: int) -> str:
    from .cache import cache_root, _key_for  # reuse repo hashing
    d = os.path.join(cache_root(), "landmarks")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_key_for(video_path)}_s{int(stride)}.npz")


# --- Per-frame memmap cache (preferred) --------------------------------------
def _lm_paths(video_path: str):
    from .cache import frames_root, _key_for
    base = os.path.join(frames_root(), _key_for(video_path))
    return base + ".landmarks.meta.json", base + ".landmarks.f32"


def _lm_lock_path(video_path: str):
    from .cache import frames_root, _key_for
    return os.path.join(frames_root(), _key_for(video_path)) + ".landmarks.lock"


def _lm_write_meta(meta_path: str, meta: dict):
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _lm_read_meta(meta_path: str):
    with open(meta_path, "r") as f:
        return json.load(f)


def ensure_landmarks_memmap(video_path: str):
    """Ensure a per-frame landmarks memmap exists and is up to date.

    - Memmap shape: (n_frames, 21, 3) float32; rows default to NaN when no hand.
    - Meta: {n, filled, src_size, src_mtime_ns, version}.
    - Builds/extends by scanning from `filled` onward using stride=1.
    """
    from .cache import _count_frames_dims, _acquire_lock, _release_lock
    meta_path, data_path = _lm_paths(video_path)
    size, mtime = _fingerprint(video_path)

    need_init = not (os.path.exists(meta_path) and os.path.exists(data_path))
    if need_init:
        n, _, _ = _count_frames_dims(video_path)
        if n <= 0:
            raise RuntimeError(f"no frames: {video_path}")
        mm = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(n, 21, 3))
        mm[:] = np.nan; mm.flush()
        _lm_write_meta(meta_path, {"n": int(n), "filled": 0, "src_size": int(size or -1), "src_mtime_ns": int(mtime or -1), "version": 1})

    # Re-open with r+; re-init if fingerprint changed or size mismatch
    meta = _lm_read_meta(meta_path)
    n = int(meta.get("n", 0))
    if n <= 0:
        os.remove(meta_path); os.remove(data_path)
        return ensure_landmarks_memmap(video_path)
    mm = np.memmap(data_path, dtype=np.float32, mode='r+', shape=(n, 21, 3))
    if int(meta.get("src_size", -1)) != int(size or -1) or int(meta.get("src_mtime_ns", -1)) != int(mtime or -1):
        mm[:] = np.nan; mm.flush()
        meta.update({"filled": 0, "src_size": int(size or -1), "src_mtime_ns": int(mtime or -1)})
        _lm_write_meta(meta_path, meta)

    # Extend if needed
    filled = int(meta.get("filled", 0))
    if filled < n:
        lock = _lm_lock_path(video_path)
        fd = _acquire_lock(lock, timeout=300.0)
        try:
            meta = _lm_read_meta(meta_path)
            filled = int(meta.get("filled", 0))
            if filled < n:
                idx, lms = _compute_landmarks_for_stride(video_path, stride=1, max_frames=0, start_from=filled)
                if len(idx) > 0:
                    mm[idx] = lms.astype(np.float32)
                mm.flush()
                meta["filled"] = n
                _lm_write_meta(meta_path, meta)
        finally:
            _release_lock(fd, lock)
    return mm, meta


def _compute_landmarks_for_stride(path: str, stride: int, max_frames: int, start_from: int = 0):
    import cv2
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for mp_logreg mode") from e
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(path)
    idx = []
    lms = []
    fi = -1
    taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi < int(start_from):
            continue
        if fi % max(1, int(stride)) != 0:
            continue
        rgb = frame[:, :, ::-1]
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in lm], dtype=float)
        idx.append(fi)
        lms.append(pts)
        taken += 1
        lim = int(max_frames)
        if lim > 0 and taken >= lim:
            break
    cap.release(); hands.close()
    if not lms:
        return np.empty((0,), dtype=int), np.empty((0, 21, 3), dtype=float)
    return np.array(idx, dtype=int), np.stack(lms, axis=0)


def pairwise_distance_features(points: np.ndarray) -> np.ndarray:
    """Return upper‑triangle (i<j) Euclidean distances as a flat vector.

    points: array shape (N, D) where N is landmark count (e.g., 21) and D=2 or 3.
    Output length is N*(N-1)/2. No fallbacks; assumes finite values.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    # Use only x,y if 3D provided (keep minimal as requested)
    pts = pts[:, :2]
    # Compute condensed distance vector (upper triangle, no diag)
    dists = []
    for i in range(n - 1):
        di = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        dists.append(di)
    v = np.concatenate(dists) if dists else np.empty(0, dtype=float)
    s = float(v.sum())
    # Normalize to an L1 distribution each time (per user request)
    if s > 0:
        v = v / s
    return v


def extract_features_for_video(path: str, stride: int = 1, max_frames: int = 0) -> np.ndarray:
    """Extract per‑frame landmark distance features from a video.

    - Uses MediaPipe Hands if available; processes every `stride` frame.
    - For each frame with a detected hand, computes pairwise distance
      features (L1‑normalized) from the first detected hand landmarks.
    - Returns an array of shape (M, F) where F=210 for 21 landmarks.

    Minimal by design; raises if mediapipe/opencv are unavailable.
    """
    # Preferred path: per-frame memmap cache (independent of stride/cap)
    try:
        mm, meta = ensure_landmarks_memmap(path)
        n = mm.shape[0]
        lim = int(max_frames)
        feats = []
        kept = 0
        for i in range(0, n, max(1, int(stride))):
            row = mm[i]
            # Skip if NaN row (no hand)
            if not np.isfinite(row).any():
                continue
            feats.append(pairwise_distance_features(row))
            kept += 1
            if lim > 0 and kept >= lim:
                break
        if feats:
            return np.stack(feats, axis=0)
    except Exception:
        pass
    # Fallback: stride-scoped cache (legacy)
    fp_size, fp_mtime = _fingerprint(path)
    cpath = _cache_path(path, stride)
    if os.path.exists(cpath):
        try:
            z = np.load(cpath)
            if int(z['src_size']) == int(fp_size) and int(z['src_mtime_ns']) == int(fp_mtime):
                idx = z['idx']; lms = z['lm']
                lim = int(max_frames)
                need_more = (lim <= 0) or (len(idx) < lim)
                if need_more:
                    start_from = int(idx[-1]) + 1 if len(idx) > 0 else 0
                    need = 0 if lim <= 0 else max(0, lim - len(idx))
                    add_idx, add_lms = _compute_landmarks_for_stride(path, stride=int(stride), max_frames=int(need), start_from=start_from)
                    if len(add_idx) > 0:
                        idx = np.concatenate([idx, add_idx])
                        lms = np.concatenate([lms, add_lms], axis=0)
                        try:
                            np.savez_compressed(cpath, idx=idx, lm=lms, src_size=int(fp_size or -1), src_mtime_ns=int(fp_mtime or -1))
                        except Exception:
                            pass
                k = len(idx) if lim <= 0 else min(len(idx), lim)
                if k > 0:
                    feats = [pairwise_distance_features(lms[i]) for i in range(k)]
                    return np.stack(feats, axis=0)
        except Exception:
            pass
    # Compute and cache
    idx, lms = _compute_landmarks_for_stride(path, stride=int(stride), max_frames=int(max_frames), start_from=0)
    if len(idx) > 0:
        try:
            sz = -1 if fp_size is None else int(fp_size)
            mt = -1 if fp_mtime is None else int(fp_mtime)
            np.savez_compressed(cpath, idx=np.asarray(idx), lm=np.asarray(lms), src_size=sz, src_mtime_ns=mt)
        except Exception:
            pass
        feats = [pairwise_distance_features(lms[i]) for i in range(len(idx))]
        return np.stack(feats, axis=0)
    return np.empty((0, 0), dtype=float)
