import sys, types


def test_apply_mjpg_mode_sets_props(monkeypatch):
    import importlib
    # Stub cv2 with minimal constants and fourcc
    class CV:
        CAP_PROP_FOURCC = 6
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        @staticmethod
        def VideoWriter_fourcc(*args):
            return 1196444237  # 'MJPG' code placeholder
    monkeypatch.setitem(sys.modules, 'cv2', CV)
    # Stub record_video._pick_first_mjpg_mode
    rv = types.SimpleNamespace(_pick_first_mjpg_mode=lambda idx: (640, 480, 30))
    monkeypatch.setitem(sys.modules, 'record_video', rv)

    inf = importlib.import_module('infer_live')

    # Fake cap recording set() calls
    calls = []
    class Cap:
        def set(self, prop, val):
            calls.append((prop, int(val)))
        def isOpened(self):
            return True
    # Force linux platform behavior
    monkeypatch.setattr(sys, 'platform', 'linux')

    cap = Cap()
    inf._apply_mjpg_mode(cap, 0)
    props = dict(calls)
    assert props.get(CV.CAP_PROP_FRAME_WIDTH) == 640
    assert props.get(CV.CAP_PROP_FRAME_HEIGHT) == 480
    assert props.get(CV.CAP_PROP_FPS) == 30

