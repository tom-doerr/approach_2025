import os
import numpy as np


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


def _compute_landmarks_for_stride(path: str, stride: int, max_frames: int):
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


def extract_features_for_video(path: str, stride: int = 5, max_frames: int = 0) -> np.ndarray:
    """Extract per‑frame landmark distance features from a video.

    - Uses MediaPipe Hands if available; processes every `stride` frame.
    - For each frame with a detected hand, computes pairwise distance
      features (L1‑normalized) from the first detected hand landmarks.
    - Returns an array of shape (M, F) where F=210 for 21 landmarks.

    Minimal by design; raises if mediapipe/opencv are unavailable.
    """
    # Try cache first (per video+stride); invalidate on source fingerprint change
    fp_size, fp_mtime = _fingerprint(path)
    cpath = _cache_path(path, stride)
    if os.path.exists(cpath):
        try:
            z = np.load(cpath)
            if int(z['src_size']) == int(fp_size) and int(z['src_mtime_ns']) == int(fp_mtime):
                idx = z['idx']; lms = z['lm']
                # Return up to max_frames worth of features (0/neg = unlimited)
                lim = int(max_frames)
                k = len(idx) if lim <= 0 else min(len(idx), lim)
                if k > 0:
                    feats = [pairwise_distance_features(lms[i]) for i in range(k)]
                    return np.stack(feats, axis=0)
        except Exception:
            pass
    # Compute and cache
    idx, lms = _compute_landmarks_for_stride(path, stride=int(stride), max_frames=int(max_frames))
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
