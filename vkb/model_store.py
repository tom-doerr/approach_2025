from __future__ import annotations

import os, pickle
from dataclasses import dataclass
from typing import Any, Iterable

from .artifacts import save_model, save_sidecar


@dataclass
class ModelStore:
    base_dir: str = "models"

    def save(self, obj: Any, name_parts: Iterable[str], *, meta: dict | None = None, tags: Iterable[str] = ("latest",), ext: str = ".pkl") -> str:
        path = save_model(obj, list(name_parts), base_dir=self.base_dir, ext=ext)
        if meta is not None:
            save_sidecar(path, meta)
        for t in (tags or ()): self.tag(path, t)
        return path

    def tag(self, model_path: str, tag: str) -> str:
        tag = str(tag).upper()
        ptr = os.path.join(self.base_dir, tag)
        os.makedirs(self.base_dir, exist_ok=True)
        with open(ptr, "w") as f:
            f.write(os.path.basename(model_path))
        return ptr

    def resolve(self, tag_or_path: str) -> str:
        p = tag_or_path
        if not os.path.isabs(p):
            # Treat as tag file if file without path
            cand = os.path.join(self.base_dir, str(tag_or_path).upper())
            if os.path.exists(cand):
                with open(cand, "r") as f:
                    bn = f.read().strip()
                p = os.path.join(self.base_dir, bn)
        return p

    def load(self, tag_or_path: str) -> Any:
        path = self.resolve(tag_or_path)
        with open(path, "rb") as f:
            return pickle.load(f)

