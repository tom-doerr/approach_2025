import json
import os
from pathlib import Path

import dspy

from dspy_agents.policy_lm import (
    configure_deepseek,
    load_aug_strength_history,
    AsyncAugPredictor,
    AugStep,
    AugSuggestion,
)


def test_configure_deepseek_sets_lm(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_key")
    lm = configure_deepseek()
    assert isinstance(lm, dspy.LM)
    # The LM should be configured globally and point at the DeepSeek chat slug
    assert getattr(dspy.settings.lm, "model", "").endswith("deepseek-chat")


def test_load_aug_strength_history_reads_meta(tmp_path: Path, monkeypatch):
    # Create two minimal sidecars with different mtimes
    m1 = tmp_path / "20250101_000000_finetune_x.pkl.meta.json"
    m2 = tmp_path / "20250102_000000_finetune_y.pkl.meta.json"
    for p, b, w in [(m1, 0.10, 0.20), (m2, 0.15, 0.25)]:
        p.write_text(json.dumps({
            "hparams": {
                "aug": "rot360",
                "brightness": b,
                "warp": w,
                "sat": 0.05,
                "contrast": 0.1,
                "hue": 0.02,
                "wb": 0.03,
                "rot_deg": 360.0
            }
        }), encoding="utf-8")
    os.utime(m1, (m1.stat().st_atime - 10, m1.stat().st_mtime - 10))

    hist = load_aug_strength_history(models_dir=str(tmp_path))
    assert len(hist) == 2
    assert hist[0]["brightness"] == 0.10 and hist[0]["warp"] == 0.20
    assert hist[1]["brightness"] == 0.15 and hist[1]["warp"] == 0.25


def test_async_predictor_runs_and_returns(monkeypatch):
    # Inject a tiny predict function that waits a bit and returns fixed values
    def fake_predict(history):
        assert isinstance(history, list) and isinstance(history[0], AugStep)
        return AugSuggestion(brightness=0.12, warp=0.22, sat=0.05, contrast=0.10, hue=0.01, wb=0.02, rot_deg=45.0)

    pred = AsyncAugPredictor(predict_fn=fake_predict)
    pred.submit([AugStep(step=1, brightness=0.1)])
    assert pred.ready() in (False, True)  # may finish fast
    out = pred.result(timeout=2.0)
    assert out["brightness"] == 0.12 and out["warp"] == 0.22
    out_t = pred.result_typed(timeout=0.5)
    assert isinstance(out_t, AugSuggestion) and out_t.brightness == 0.12
