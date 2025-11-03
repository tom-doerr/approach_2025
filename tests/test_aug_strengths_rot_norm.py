import types


def test_aug_strengths_rot_normalized_to_one():
    from vkb.finetune import _epoch_after

    # Fake console and telemetry to capture scalar2 calls
    class Cons:
        def print(self, *a, **k):
            return

    calls = []
    class Tel:
        def __init__(self): pass
        def metric(self, *a, **k): pass
        def scalar(self, *a, **k): pass
        def text(self, *a, **k): pass
        def scalar2(self, win_key, series, step, value, **_):
            calls.append((win_key, series, step, float(value)))

    # args with dynamic_aug True and a train dataset exposing aug strengths
    ds = types.SimpleNamespace(brightness=0.1, warp=0.2, sat=0.0, contrast=0.0, hue=0.0, wb=0.0, rot_deg=360.0)
    args = types.SimpleNamespace(dynamic_aug=True, _ds_tr=ds, epochs=1, aug='none', brightness=0.0, warp=0.0, drop_path=0.0, dropout=0.0)

    _epoch_after(Cons(), Tel(), 1, args, tr_acc=0.5, val_acc=0.5)

    # Find the rotation series and assert it logs explicit label and value in degrees
    rot = [c for c in calls if c[1] == 'rotation_deg']
    assert rot, 'rotation_deg series not logged'
    assert abs(rot[0][3] - 360.0) < 1e-6
