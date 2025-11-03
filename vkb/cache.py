import os
import numpy as np
import json
from dataclasses import dataclass


def embed_id(model_name: str, embed_params: dict | None = None) -> str:
    return model_name


def cache_root() -> str:
    return os.path.join(".cache", "vkb")


def cache_dir(model_name: str, embed_params: dict | None = None) -> str:
    d = os.path.join(cache_root(), embed_id(model_name, embed_params))
    os.makedirs(d, exist_ok=True)
    return d


def cache_path_for(cache_dir: str, data_root: str, video_path: str) -> str:
    rel = os.path.relpath(video_path, data_root)
    p = os.path.join(cache_dir, rel + ".npy")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_embeddings(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def load_embeddings(path: str):
    return np.load(path) if os.path.exists(path) else None


# --- Frame cache (for video random access) ---
def frames_root() -> str:
    d = os.path.join(cache_root(), "frames")
    os.makedirs(d, exist_ok=True)
    return d


def _key_for(video_path: str) -> str:
    import hashlib
    ap = os.path.abspath(video_path)
    return hashlib.sha1(ap.encode("utf-8")).hexdigest()[:16]


def frames_paths(data_root: str, video_path: str):
    base = os.path.join(frames_root(), _key_for(video_path))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    return base + ".meta.json", base + ".uint8"


def _lock_path(video_path: str) -> str:
    return os.path.join(frames_root(), _key_for(video_path)) + ".lock"


def _lock_ttl_sec() -> int:
    try:
        return int(os.getenv("VKB_CACHE_LOCK_TTL_SEC", "3600"))
    except Exception:
        return 3600


def _is_lock_stale(lock_path: str) -> bool:
    import time
    try:
        # mtime-based TTL
        ttl = _lock_ttl_sec()
        if ttl > 0:
            age = time.time() - os.path.getmtime(lock_path)
            if age > ttl:
                return True
        # PID liveness (best-effort)
        try:
            with open(lock_path, "rb") as f:
                txt = f.read().strip()[:16]
            pid = int(txt) if txt else None
        except Exception:
            pid = None
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            except Exception:
                pass
    except Exception:
        pass
    return False


def _acquire_lock(lock_path: str, timeout: float = 60.0):
    import time, os
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            # Clear obviously stale locks to avoid multi-minute stalls/spin
            if _is_lock_stale(lock_path):
                try:
                    os.unlink(lock_path)
                    if os.getenv('VKB_CACHE_LOG',''):
                        print(f"cache: cleared stale lock {os.path.basename(lock_path)}")
                except Exception:
                    pass
            if time.time() - start > timeout:
                raise TimeoutError(f"timeout waiting for lock: {lock_path}")
            time.sleep(0.05)


def _release_lock(fd: int, lock_path: str):
    import os
    try:
        os.close(fd)
    finally:
        try:
            os.unlink(lock_path)
        except Exception:
            pass


def _src_fingerprint(video_path: str):
    """Return (size_bytes, mtime_ns) or (None, None) if file not found."""
    src_size = src_mtime = None
    if os.path.exists(video_path):
        st = os.stat(video_path)
        src_size = int(st.st_size)
        src_mtime = int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)))
    return src_size, src_mtime


def _check_existing(meta_path: str, data_path: str, video_path: str, src_size, src_mtime):
    """Validate existing cache; return (meta or None, ok: bool, why: str|None)."""
    if not (os.path.exists(meta_path) and os.path.exists(data_path)):
        return None, False, None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        exp_bytes = int(meta["n"]) * int(meta["h"]) * int(meta["w"]) * 3
        size_ok = os.path.getsize(data_path) == exp_bytes
        import cv2 as _cv
        cap0 = _cv.VideoCapture(video_path)
        ok0, fr0 = cap0.read(); cap0.release()
        dims_ok = ok0 and fr0.shape[0] == int(meta["h"]) and fr0.shape[1] == int(meta["w"]) 
        fp_ok = True
        if src_size is not None and src_mtime is not None:
            fp_ok = (int(meta.get("src_size", -1)) == src_size) and (int(meta.get("src_mtime_ns", -1)) == src_mtime)
        if size_ok and dims_ok and fp_ok:
            return meta, True, None
        bad = []
        if not size_ok: bad.append("size")
        if not dims_ok: bad.append("dims")
        if not fp_ok: bad.append("fingerprint")
        return None, False, ",".join(bad) if bad else "invalid"
    except Exception:
        return None, False, None


def _recheck_after_wait(meta_path: str, data_path: str, video_path: str, src_size, src_mtime):
    """Re-validate after waiting for a builder; same return contract as _check_existing."""
    return _check_existing(meta_path, data_path, video_path, src_size, src_mtime)


def _count_frames_dims(video_path: str):
    import cv2 as cv
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, None, None
    count = 0; h = w = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if h is None:
            h, w = fr.shape[:2]
        count += 1
    cap.release()
    return count, h, w


def _build_frames_cache(video_path: str, data_path: str, meta_path: str, h: int, w: int, count: int, src_size, src_mtime, console=None, why: str | None = None):
    import numpy as np
    import cv2 as cv
    mm = np.memmap(data_path, dtype=np.uint8, mode='w+', shape=(count, h, w, 3))
    cap = cv.VideoCapture(video_path); i = 0
    while i < count:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[0] != h or fr.shape[1] != w:
            break
        mm[i] = fr; i += 1
    cap.release(); mm.flush()
    meta = {"n": int(i), "h": int(h), "w": int(w)}
    if src_size is not None and src_mtime is not None:
        meta["src_size"] = src_size
        meta["src_mtime_ns"] = src_mtime
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta


def ensure_frames_cached(data_root: str, video_path: str, console=None):
    import cv2 as cv
    meta_path, data_path = frames_paths(data_root, video_path)
    lock_path = _lock_path(video_path)
    # Source fingerprint
    src_size, src_mtime = _src_fingerprint(video_path)
    why = None
    meta, ok, why0 = _check_existing(meta_path, data_path, video_path, src_size, src_mtime)
    if ok:
        return meta, False
    if why0:
        why = why0
    # If another worker is building, wait for the lock to clear, then re-check
    if os.path.exists(lock_path):
        try:
            fd_wait = _acquire_lock(lock_path, timeout=300.0)  # block until free, then acquire
        except TimeoutError:
            raise
        else:
            _release_lock(fd_wait, lock_path)  # immediately release; another worker already built
        # re-check after waiting
        meta, ok, why1 = _recheck_after_wait(meta_path, data_path, video_path, src_size, src_mtime)
        if ok:
            return meta, False
        if why1:
            why = why1
    # First pass: count frames and collect dims
    # Acquire build lock
    fd = _acquire_lock(lock_path, timeout=300.0)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        _release_lock(fd, lock_path)
        raise RuntimeError(f"cannot open video: {video_path}")
    cap.release()  # recount using helper to keep code small
    count, h, w = _count_frames_dims(video_path)
    if count == 0 or h is None:
        _release_lock(fd, lock_path)
        raise RuntimeError(f"no frames: {video_path}")
    # Second pass: write memmap sequentially
    try:
        # Optional one-line notice when rebuilding
        try:
            if why is None:
                msg = f"cache build: {os.path.basename(video_path)}"
            else:
                msg = f"cache rebuild: {os.path.basename(video_path)} reason={why}"
            if console is not None:
                try:
                    console.print(f"[dim]{msg}[/]")
                except Exception:
                    pass
            else:
                import os as _os
                if _os.getenv('VKB_CACHE_LOG',''):
                    print(msg)
        except Exception:
            pass
        meta = _build_frames_cache(video_path, data_path, meta_path, h, w, count, src_size, src_mtime, console=console, why=why)
        return meta, True
    finally:
        _release_lock(fd, lock_path)


def open_frames_memmap(data_root: str, video_path: str):
    meta_path, data_path = frames_paths(data_root, video_path)
    with open(meta_path, "r") as f:
        m = json.load(f)
    n, h, w = int(m["n"]), int(m["h"]), int(m["w"])
    mm = np.memmap(data_path, dtype=np.uint8, mode='r', shape=(n, h, w, 3))
    return mm, m


# --- Minimal OO wrapper -------------------------------------------------------
@dataclass
class Cache:
    model_name: str = "mobilenetv3_small_100"
    data_root: str = "data"
    embed_params: dict | None = None

    # Embedding cache
    def dir(self) -> str:
        return cache_dir(self.model_name, self.embed_params)

    def path_for(self, video_path: str) -> str:
        return cache_path_for(self.dir(), self.data_root, video_path)

    def load(self, video_path: str):
        return load_embeddings(self.path_for(video_path))

    def save(self, video_path: str, arr: np.ndarray) -> str:
        p = self.path_for(video_path)
        save_embeddings(p, arr)
        return p

    # Frame cache
    def ensure_frames(self, video_path: str, console=None):
        return ensure_frames_cached(self.data_root, video_path, console)

    def open_frames(self, video_path: str):
        return open_frames_memmap(self.data_root, video_path)
