import os, time
from pathlib import Path


def read_pid(path: str) -> int | None:
    try:
        with open(path, "rb") as f:
            return int(f.read().strip() or b"0")
    except Exception:
        return None


def test_acquire_lock_clears_stale_by_ttl(monkeypatch, tmp_path: Path):
    from vkb.cache import _acquire_lock, _release_lock
    lock = tmp_path / "vid.lock"
    lock.write_text("12345")
    # Force low TTL so this lock is stale
    monkeypatch.setenv("VKB_CACHE_LOCK_TTL_SEC", "1")
    old = time.time() - 5
    os.utime(lock, (old, old))
    fd = _acquire_lock(str(lock), timeout=1.0)
    try:
        assert lock.exists()
        pid = read_pid(str(lock))
        assert pid is not None and pid == os.getpid()
    finally:
        _release_lock(fd, str(lock))


def test_acquire_lock_clears_dead_pid(monkeypatch, tmp_path: Path):
    from vkb.cache import _acquire_lock, _release_lock
    lock = tmp_path / "vid.lock"
    lock.write_text("999999")  # very unlikely to exist
    # Set a very large TTL so only PID liveness triggers
    monkeypatch.setenv("VKB_CACHE_LOCK_TTL_SEC", "999999")
    fd = _acquire_lock(str(lock), timeout=1.0)
    try:
        assert lock.exists()
        pid = read_pid(str(lock))
        assert pid is not None and pid == os.getpid()
    finally:
        _release_lock(fd, str(lock))

