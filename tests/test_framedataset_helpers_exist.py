def test_framedataset_helpers_exist():
    from vkb.dataset import FrameDataset
    ds = FrameDataset([("v",0,0)], img_size=8)
    for name in ("_open_memmap","_apply_xy_shift","_apply_rotation","_apply_color_and_geom_augs"):
        assert hasattr(ds, name), f"missing helper: {name}"

