import os
from vkb.artifacts import save_model, list_models, latest_model


def test_artifacts_respects_env_tmpdir(tmp_path, monkeypatch):
    base = tmp_path / "models_env"
    monkeypatch.setenv("VKB_MODELS_DIR", str(base))
    p = save_model({"x": 1}, ["unit", "test"], base_dir="models")
    assert str(base) in p
    # latest and list should also read from env dir
    lst = list_models()
    assert lst and all(str(base) in x for x in lst)
    assert str(base) in latest_model()

