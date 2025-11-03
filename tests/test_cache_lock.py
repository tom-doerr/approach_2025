import os, time, threading
from vkb.cache import frames_root, _acquire_lock, _release_lock


def test_lock_acquire_blocks_until_release(tmp_path):
    # Create a unique lock path under frames cache to avoid collisions
    lock_path = os.path.join(frames_root(), f"test_{int(time.time()*1e6)}.lock")

    acquired = []

    def holder():
        fd = _acquire_lock(lock_path, timeout=1.0)
        acquired.append(fd)
        time.sleep(0.2)
        _release_lock(fd, lock_path)

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.05)  # let holder acquire

    # While holder has the lock, trying to acquire with a tiny timeout should fail
    err = None
    try:
        _acquire_lock(lock_path, timeout=0.05)
    except TimeoutError as e:
        err = e
    assert err is not None

    # After release, we should be able to acquire
    t.join()
    fd2 = _acquire_lock(lock_path, timeout=0.5)
    _release_lock(fd2, lock_path)
    assert True

