import os, pickle, glob, json
from dataclasses import dataclass
from datetime import datetime
from .io import safe_label


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _basedir(d: str) -> str:
    return os.getenv('VKB_MODELS_DIR', d)


def save_model(obj, name_parts, base_dir="models", ext=".pkl"):
    base_dir = _basedir(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    name = "_".join(safe_label(str(p)) for p in name_parts)
    path = os.path.join(base_dir, f"{timestamp()}_{name}{ext}")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    # Write pointer to latest for robust default selection
    try:
        latest_ptr = os.path.join(base_dir, "LATEST")
        with open(latest_ptr, "w") as lf:
            lf.write(os.path.basename(path))
    except Exception:
        pass
    return path

def path_for(name_parts, base_dir="models", ext=".pkl", ts: str | None = None):
    """Build a deterministic model path (does not write).

    Useful for printing a stable path per run/epoch. If `ts` is
    provided, it is used as the timestamp prefix; otherwise, current
    time is used.
    """
    base_dir = _basedir(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    name = "_".join(safe_label(str(p)) for p in name_parts)
    t = ts or timestamp()
    return os.path.join(base_dir, f"{t}_{name}{ext}")

def save_model_at(obj, path: str):
    """Write model to an explicit `path` and update LATEST pointer."""
    base_dir = _basedir(os.path.dirname(path) or "models")
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    try:
        latest_ptr = os.path.join(base_dir, "LATEST")
        with open(latest_ptr, "w") as lf:
            lf.write(os.path.basename(path))
    except Exception:
        pass
    return path


def list_models(base_dir="models", pattern="*.pkl"):
    base_dir = _basedir(base_dir)
    paths = glob.glob(os.path.join(base_dir, pattern))
    return sorted(paths)


def latest_model(base_dir="models", pattern="*.pkl"):
    base_dir = _basedir(base_dir)
    # Prefer explicit pointer if present
    ptr = os.path.join(base_dir, "LATEST")
    if os.path.exists(ptr):
        try:
            with open(ptr, "r") as f:
                bn = f.read().strip()
            # Allow absolute path pointers for cross-dir cases
            cand = bn if os.path.isabs(bn) else os.path.join(base_dir, bn)
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
    paths = list_models(base_dir, pattern)
    if not paths:
        raise FileNotFoundError(f"no models found in {base_dir}")
    return paths[-1]


def save_sidecar(model_path: str, meta: dict):
    """Write a small JSON sidecar with run metadata next to `model_path`.

    File name: `<model>.meta.json` in the same directory.
    """
    d = os.path.dirname(model_path)
    bn = os.path.basename(model_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    out = os.path.join(d, bn + ".meta.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return out


# Minimal OO wrapper for artifacts
@dataclass
class Artifacts:
    base_dir: str = "models"
    pattern: str = "*.pkl"

    def save(self, obj, name_parts, ext: str = ".pkl") -> str:
        return save_model(obj, name_parts, base_dir=self.base_dir, ext=ext)

    def list(self, pattern: str | None = None):
        return list_models(base_dir=self.base_dir, pattern=(pattern or self.pattern))

    def latest(self, pattern: str | None = None):
        return latest_model(base_dir=self.base_dir, pattern=(pattern or self.pattern))

    def save_sidecar(self, model_path: str, meta: dict) -> str:
        return save_sidecar(model_path, meta)
