from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class BullseyeDetection:
    pixel_x: float
    pixel_y: float
    score: float
    x_norm: float
    y_norm: float

    def to_pixel_list(self) -> list[float]:
        return [self.pixel_x, self.pixel_y]


class BullseyeDetector(Protocol):
    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        ...


class BullseyeVision(ABC):
    @abstractmethod
    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        raise NotImplementedError


def build_detection(
    *,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    score: float,
) -> BullseyeDetection:
    safe_width = max(int(width), 1)
    safe_height = max(int(height), 1)
    clipped_x = float(np.clip(float(pixel_x), 0.0, float(safe_width) - 1.0))
    clipped_y = float(np.clip(float(pixel_y), 0.0, float(safe_height) - 1.0))
    return BullseyeDetection(
        pixel_x=clipped_x,
        pixel_y=clipped_y,
        score=float(score),
        x_norm=clipped_x / float(safe_width),
        y_norm=clipped_y / float(safe_height),
    )
