from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import onnxruntime as ort

from target_lock.vision.base import BullseyeDetection, BullseyeVision


class CvBullseyeVision(BullseyeVision):
    def __init__(
        self,
        *,
        onnx_path: str | Path,
        img_size_fallback: int = 640,
        score_threshold: float = 0.0,
    ) -> None:
        self.onnx_path = Path(onnx_path).expanduser()
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"YOLO onnx model not found: {self.onnx_path}")
        self.score_threshold = float(score_threshold)
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = self.resolve_img_size(self.session, img_size_fallback)

    @staticmethod
    def letterbox_frame(
        image_rgb: np.ndarray,
        size: int,
        pad_value: int = 114,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        height, width = image_rgb.shape[:2]
        scale = min(size / height, size / width)
        resized_height = int(round(height * scale))
        resized_width = int(round(width * scale))
        resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
        pad_w = (size - resized_width) // 2
        pad_h = (size - resized_height) // 2
        canvas[pad_h:pad_h + resized_height, pad_w:pad_w + resized_width] = resized
        return canvas, {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_w": width,
            "orig_h": height,
        }

    def preprocess_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
        resized, meta = self.letterbox_frame(frame_rgb, self.img_size)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, meta

    def postprocess_point(
        self,
        point_xy: np.ndarray,
        meta: dict[str, float | int],
    ) -> tuple[float, float, float, float]:
        x_model = float(point_xy[0]) * self.img_size
        y_model = float(point_xy[1]) * self.img_size

        x_orig = (x_model - float(meta["pad_w"])) / float(meta["scale"])
        y_orig = (y_model - float(meta["pad_h"])) / float(meta["scale"])

        x_orig = float(np.clip(x_orig, 0.0, float(meta["orig_w"]) - 1.0))
        y_orig = float(np.clip(y_orig, 0.0, float(meta["orig_h"]) - 1.0))

        x_norm = x_orig / float(meta["orig_w"])
        y_norm = y_orig / float(meta["orig_h"])
        return x_orig, y_orig, x_norm, y_norm

    @staticmethod
    def resolve_img_size(session, fallback: int) -> int:
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) >= 4:
            height = input_shape[2]
            width = input_shape[3]
            if isinstance(height, int) and isinstance(width, int) and height == width:
                return int(height)
        return fallback

    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        del info
        tensor, meta = self.preprocess_frame(frame_rgb)
        points, scores = self.session.run(None, {self.input_name: tensor})

        point_xy = points[0]
        score = float(scores[0, 0])
        if score < self.score_threshold:
            return None

        x_orig, y_orig, x_norm, y_norm = self.postprocess_point(point_xy, meta)
        return BullseyeDetection(
            pixel_x=x_orig,
            pixel_y=y_orig,
            score=score,
            x_norm=x_norm,
            y_norm=y_norm,
        )
