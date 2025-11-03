from __future__ import annotations

from typing import Any, Dict, List, Callable, Optional, Iterable
from dataclasses import dataclass, asdict
import json
import os
import threading
from glob import glob

import dspy


@dataclass
class PolicyStep:
    step: int
    # Augmentation knobs
    brightness: float | None = None
    warp: float | None = None
    sat: float | None = None
    contrast: float | None = None
    hue: float | None = None
    wb: float | None = None
    rot_deg: float | None = None
    # Training knobs/metrics
    dropout: float | None = None
    drop_path: float | None = None
    train_acc: float | None = None
    train_loss: float | None = None
    val_acc: float | None = None
    val_loss: float | None = None


@dataclass
class PolicySuggestion:
    brightness: float | None = None
    warp: float | None = None
    sat: float | None = None
    contrast: float | None = None
    hue: float | None = None
    wb: float | None = None
    rot_deg: float | None = None
    dropout: float | None = None
    drop_path: float | None = None


def configure_deepseek(model: str = "openai/deepseek-chat", api_key: str | None = None, api_base: str = "https://api.deepseek.com") -> dspy.LM:
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    lm = dspy.LM(model, api_key=key, api_base=api_base)
    dspy.configure(lm=lm)
    return lm


def configure_openrouter(model: str = "openrouter/deepseek-chat", api_key: str | None = None, api_base: str = "https://openrouter.ai/api/v1", reasoning_effort: str | None = None) -> dspy.LM:
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    kwargs = {"reasoning": {"effort": str(reasoning_effort)}} if reasoning_effort else {}
    lm = dspy.LM(model, api_key=key, api_base=api_base, **kwargs)
    dspy.configure(lm=lm)
    return lm


def load_aug_strength_history(models_dir: str | None = None) -> List[Dict[str, Any]]:
    root = models_dir or os.getenv("VKB_MODELS_DIR", "models")
    paths = sorted(glob(os.path.join(root, "*.meta.json")), key=lambda p: os.path.getmtime(p))
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(paths, 1):
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        hp = meta.get("hparams", {})
        out.append({
            "step": i,
            "path": p,
            "aug": hp.get("aug", "none"),
            "brightness": float(hp.get("brightness", 0.0) or 0.0),
            "warp": float(hp.get("warp", 0.0) or 0.0),
            "sat": float(hp.get("sat", 0.0) or 0.0),
            "contrast": float(hp.get("contrast", 0.0) or 0.0),
            "hue": float(hp.get("hue", 0.0) or 0.0),
            "wb": float(hp.get("wb", 0.0) or 0.0),
            "rot_deg": float(hp.get("rot_deg", 0.0) or 0.0),
        })
    return out


def _history_to_text(history: List[Dict[str, Any]], k: int = 8) -> str:
    if not history:
        return "(no history)"
    tail = history[-k:]
    lines = []
    for h in tail:
        step = h.get("step", "?")
        vals = [
            f"b={h.get('brightness', 0.0):.3f}", f"w={h.get('warp', 0.0):.3f}", f"s={h.get('sat', 0.0):.3f}",
            f"c={h.get('contrast', 0.0):.3f}", f"hue={h.get('hue', 0.0):.3f}", f"wb={h.get('wb', 0.0):.3f}", f"rot={h.get('rot_deg', 0.0):.1f}",
        ]
        if h.get("dropout") is not None:
            vals.append(f"do={h.get('dropout'):.3f}")
        if h.get("drop_path") is not None:
            vals.append(f"dp={h.get('drop_path'):.3f}")
        va = h.get("val_acc")
        if va is not None:
            vals.append(f"val={float(va):.4f}")
        lines.append(f"t{step}: " + " ".join(vals))
    return "\n".join(lines)


class _SuggestPolicySig(dspy.Signature):
    """ 
            Optimize the validation accuracy. The history contains the values you previously used.
            Please experiment and don't do just small changes every step.
            It's okay to overshoot sometimes, you probably should overshoot to figure out what the currently best value is.
            Do not change all parameters the same way, change them by different percentages and/or directions so you get more feedback on which parameter values has what influence currently.
            I do think we should keep train accuracy well under a perfect 1.0 accuracy so the model doesn't overfit and is actually learning useful features.
            Agressively regularize and augment if train accuracy is near 1.0; there are a lot of near duplicate images in the data.
    """
    history = dspy.InputField()
    suggestion_json = dspy.OutputField()


def _coerce_steps(history: Iterable[Dict[str, Any]] | Iterable[PolicyStep]) -> List[PolicyStep]:
    out: List[PolicyStep] = []
    for h in history:
        if isinstance(h, PolicyStep):
            out.append(h)
        else:
            out.append(PolicyStep(
                step=int(h.get("step", 0)),
                brightness=h.get("brightness"), warp=h.get("warp"), sat=h.get("sat"), contrast=h.get("contrast"), hue=h.get("hue"), wb=h.get("wb"), rot_deg=h.get("rot_deg"),
                dropout=h.get("dropout"), drop_path=h.get("drop_path"),
                train_acc=h.get("train_acc"), train_loss=h.get("train_loss"), val_acc=h.get("val_acc"), val_loss=h.get("val_loss"),
            ))
    return out


def suggest_policy_typed(history: Iterable[Dict[str, Any]] | Iterable[PolicyStep]) -> PolicySuggestion:
    if dspy.settings.lm is None:
        raise RuntimeError("DSPy LM is not configured. Call configure_deepseek()/configure_openrouter() first.")
    steps = _coerce_steps(history)
    prompt = _history_to_text([asdict(s) for s in steps])
    predictor = dspy.Predict(_SuggestPolicySig)
    out = predictor(history=(
        "You tune training policy (augs + regularization). Respond with STRICT JSON only.\n"
        "Keys (optional): brightness, warp, sat, contrast, hue, wb, rot_deg, dropout, drop_path.\n"
        "Ranges: brightness[0..0.6], warp[0..0.6], sat[0..0.8], contrast[0..0.8], hue[0..0.5], wb[0..0.5], rot_deg[0..360], dropout[0..0.8], drop_path[0..0.8].\n"
        "Prefer small changes from the last step.\n\nHistory:\n" + prompt
    ))
    data = json.loads(str(out.suggestion_json))
    return PolicySuggestion(
        brightness=data.get("brightness"), warp=data.get("warp"), sat=data.get("sat"), contrast=data.get("contrast"), hue=data.get("hue"), wb=data.get("wb"), rot_deg=data.get("rot_deg"),
        dropout=data.get("dropout"), drop_path=data.get("drop_path"),
    )


def suggest_policy_sync(history: Iterable[Dict[str, Any]] | Iterable[PolicyStep]) -> Dict[str, float]:
    s = suggest_policy_typed(history)
    return {k: v for k, v in asdict(s).items() if v is not None}


@dataclass
class AsyncPolicyPredictor:
    predict_fn: Optional[Callable[[List[PolicyStep]], PolicySuggestion | Dict[str, float]]] = None

    def __post_init__(self):
        self._thr: Optional[threading.Thread] = None
        self._event = threading.Event()
        self._result: Optional[Dict[str, float] | PolicySuggestion] = None
        self._err: Optional[BaseException] = None

    def submit(self, history: List[Dict[str, Any]] | List[PolicyStep]):
        def _run():
            try:
                fn = self.predict_fn or suggest_policy_typed
                steps = _coerce_steps(history)
                res = fn(steps)
                self._result = res
            except BaseException as e:
                self._err = e
            finally:
                self._event.set()
        if self._thr and self._thr.is_alive():
            raise RuntimeError("prediction already in progress")
        self._event.clear(); self._err = None; self._result = None
        self._thr = threading.Thread(target=_run, daemon=True)
        self._thr.start()

    def ready(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> Dict[str, float]:
        if not self._event.wait(timeout):
            raise TimeoutError("prediction not ready")
        if self._err:
            raise self._err
        assert self._result is not None
        if isinstance(self._result, PolicySuggestion):
            return {k: v for k, v in asdict(self._result).items() if v is not None}
        return self._result

    def result_typed(self, timeout: Optional[float] = None) -> PolicySuggestion:
        if not self._event.wait(timeout):
            raise TimeoutError("prediction not ready")
        if self._err:
            raise self._err
        assert self._result is not None
        if isinstance(self._result, PolicySuggestion):
            return self._result
        d = self._result
        return PolicySuggestion(**d)


# Backward-compatible aliases (Aug* names)
AugStep = PolicyStep
AugSuggestion = PolicySuggestion
AsyncAugPredictor = AsyncPolicyPredictor
suggest_aug_typed = suggest_policy_typed
suggest_aug_sync = suggest_policy_sync


def stream_reasoning_openrouter(
    steps: Iterable[PolicyStep] | Iterable[Dict[str, Any]],
    model: str = "deepseek/deepseek-reasoner",
    api_key: Optional[str] = None,
    effort: Optional[str] = None,
    on_delta: Optional[Callable[[str], None]] = None,
    api_base: str = "https://openrouter.ai/api/v1",
):
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    import requests, sseclient
    hist = _coerce_steps(steps)
    prompt = _history_to_text([asdict(s) for s in hist])
    url = f"{api_base.rstrip('/')}/responses"
    body: Dict[str, Any] = {
        "model": model,
        "input": f"You are tuning training policy. Think step-by-step.\nHistory:\n{prompt}\n",
        "stream": True,
    }
    if effort:
        body["reasoning"] = {"effort": str(effort)}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    with requests.post(url, headers=headers, json=body, stream=True, timeout=60) as r:
        r.raise_for_status()
        client = sseclient.SSEClient(r)
        for ev in client.events():
            if not ev.data or ev.data == "[DONE]":
                continue
            try:
                obj = json.loads(ev.data)
            except Exception:
                continue
            t = obj.get("type") or obj.get("event")
            if t == "response.reasoning.delta":
                delta = obj.get("delta", "")
                if delta and on_delta:
                    try: on_delta(delta)
                    except Exception: pass

