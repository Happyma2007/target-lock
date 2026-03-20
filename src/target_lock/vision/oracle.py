from __future__ import annotations

from typing import Mapping

import numpy as np

from target_lock.vision.base import BullseyeDetection, BullseyeVision, build_detection


class OracleBullseyeVision(BullseyeVision):
    def __init__(self, *, score: float = 1.0) -> None:
        self.score = float(score)

    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        if info is None:
            return None

        bullseye_pixel = info.get("bullseye_pixel")
        if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
            return None

        height, width = frame_rgb.shape[:2]
        width = int(info.get("width", width))
        height = int(info.get("height", height))
        return build_detection(
            pixel_x=float(bullseye_pixel[0]),
            pixel_y=float(bullseye_pixel[1]),
            width=width,
            height=height,
            score=self.score,
        )
