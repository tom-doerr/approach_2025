from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    def __call__(self, frame: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class FrameCache(Protocol):
    def ensure_frames_cached(self, data_root: str, video_path: str) -> None: ...
    def open_frames_memmap(self, data_root: str, video_path: str) -> tuple[np.ndarray, dict]: ...


@runtime_checkable
class Augmenter(Protocol):
    def apply(self, img: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Console(Protocol):
    def print(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class Classifier(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def score(self, X, y) -> float: ...

