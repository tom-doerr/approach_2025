import sys, types, os
from types import SimpleNamespace


def test_cap_threads_sets_env_and_calls_setters(monkeypatch):
    # Stub cv2 and torch with minimal APIs
    cv_calls = {}
    torch_calls = {}

    cv2 = types.SimpleNamespace()
    def _cv_set(n):
        cv_calls['set'] = n
    cv2.setNumThreads = _cv_set
    monkeypatch.setitem(sys.modules, 'cv2', cv2)

    torch = types.SimpleNamespace()
    def _t_set(n):
        torch_calls['set'] = n
    def _t_seti(n):
        torch_calls['seti'] = n
    torch.set_num_threads = _t_set
    torch.set_num_interop_threads = _t_seti
    monkeypatch.setitem(sys.modules, 'torch', torch)

    # Import target after stubbing
    import importlib
    tf = importlib.import_module('train_frames')

    # Clear env to known state
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.pop(k, None)

    args = SimpleNamespace(cap_threads=True)
    tf._apply_thread_caps(args)

    assert os.environ.get('OMP_NUM_THREADS') == '1'
    assert os.environ.get('MKL_NUM_THREADS') == '1'
    assert os.environ.get('OPENBLAS_NUM_THREADS') == '1'
    assert os.environ.get('NUMEXPR_NUM_THREADS') == '1'
    assert cv_calls.get('set') == 1
    assert torch_calls.get('set') == 1
    assert torch_calls.get('seti') == 1

def test_cap_threads_noop_when_flag_false(monkeypatch):
    # Ensure it doesn't explode or set env when disabled
    import importlib
    tf = importlib.import_module('train_frames')
    from types import SimpleNamespace
    args = SimpleNamespace(cap_threads=False)
    tf._apply_thread_caps(args)
    # No assertions needed; just ensure no exceptions
    assert True

