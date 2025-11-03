from __future__ import annotations

from typing import Any
import numpy as np
from vkb.ports import FrameCache


class CacheModuleAdapter(FrameCache):
    def __init__(self, module: Any):
        self._m = module

    def ensure_frames_cached(self, data_root: str, video_path: str) -> None:
        self._m.ensure_frames_cached(data_root, video_path)

    def open_frames_memmap(self, data_root: str, video_path: str) -> tuple[np.ndarray, dict]:
        return self._m.open_frames_memmap(data_root, video_path)

