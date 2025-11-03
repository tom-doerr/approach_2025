import os, json, tempfile


def test_artifacts_wrapper_roundtrip():
    from vkb.artifacts import Artifacts
    with tempfile.TemporaryDirectory() as tmp:
        store = Artifacts(base_dir=os.path.join(tmp, "models"))
        obj = {"clf_name": "test", "labels": ["A","B"]}
        path = store.save(obj, ["finetune", "mobilenetv3_small_100"])
        assert os.path.exists(path)
        listed = store.list()
        assert path in listed
        latest = store.latest()
        assert latest == path
        meta_path = store.save_sidecar(path, {"val_acc": 0.5})
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta.get("val_acc") == 0.5

