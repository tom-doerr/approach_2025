import numpy as np


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


def extract_features_for_video(path: str, stride: int = 5, max_frames: int = 200) -> np.ndarray:
    """Extract per‑frame landmark distance features from a video.

    - Uses MediaPipe Hands if available; processes every `stride` frame.
    - For each frame with a detected hand, computes pairwise distance
      features (L1‑normalized) from the first detected hand landmarks.
    - Returns an array of shape (M, F) where F=210 for 21 landmarks.

    Minimal by design; raises if mediapipe/opencv are unavailable.
    """
    import cv2  # noqa: F401 (import locally)
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for mp_logreg mode") from e

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(path)
    feats = []
    count = 0
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % max(1, int(stride)) != 0:
            continue
        rgb = frame[:, :, ::-1]
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in lm], dtype=float)
        feats.append(pairwise_distance_features(pts))
        count += 1
        if count >= int(max_frames):
            break
    cap.release()
    hands.close()
    return np.stack(feats, axis=0) if feats else np.empty((0, 0), dtype=float)

