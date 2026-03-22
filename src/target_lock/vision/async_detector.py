from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Callable, Mapping

import numpy as np

from target_lock.vision.base import BullseyeDetection, BullseyeDetector, build_detection
from target_lock.vision.cv import CvBullseyeVision


@dataclass(slots=True)
class _PendingRequest:
    frame_rgb: np.ndarray
    info: dict[str, object]


class AsyncCvBullseyeVision:
    def __init__(
        self,
        *,
        onnx_path: str | Path,
        img_size_fallback: int = 640,
        score_threshold: float = 0.0,
        smoothing_alpha: float = 1.0,
        detector_factory: Callable[[], BullseyeDetector] | None = None,
    ) -> None:
        if not 0.0 < float(smoothing_alpha) <= 1.0:
            raise ValueError("smoothing_alpha must be in (0.0, 1.0]")

        self.onnx_path = Path(onnx_path).expanduser()
        self.img_size_fallback = int(img_size_fallback)
        self.score_threshold = float(score_threshold)
        self.smoothing_alpha = float(smoothing_alpha)
        self._detector_factory = detector_factory or self._build_detector

        self._condition = Condition()
        self._ready = Event()
        self._pending_request: _PendingRequest | None = None
        self._latest_detection: BullseyeDetection | None = None
        self._closed = False
        self._worker_exception: BaseException | None = None
        self._thread = Thread(
            target=self._worker_main,
            name="target-lock-vision",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        self._raise_if_worker_failed()

    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        self._raise_if_worker_failed()
        request = _PendingRequest(
            frame_rgb=np.ascontiguousarray(frame_rgb).copy(),
            info=dict(info or {}),
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("AsyncCvBullseyeVision is closed")
            current = self._latest_detection
            self._pending_request = request
            self._condition.notify()
            return current

    def get_latest_detection(self) -> BullseyeDetection | None:
        self._raise_if_worker_failed()
        with self._condition:
            return self._latest_detection

    def reset(self) -> None:
        self._raise_if_worker_failed()
        with self._condition:
            self._pending_request = None
            self._latest_detection = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending_request = None
            self._condition.notify_all()
        self._thread.join()

    def _build_detector(self) -> BullseyeDetector:
        return CvBullseyeVision(
            onnx_path=self.onnx_path,
            img_size_fallback=self.img_size_fallback,
            score_threshold=self.score_threshold,
        )

    def _worker_main(self) -> None:
        detector: BullseyeDetector | None = None
        try:
            detector = self._detector_factory()
            self._ready.set()
            while True:
                with self._condition:
                    while self._pending_request is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    request = self._pending_request
                    self._pending_request = None
                    previous = self._latest_detection

                detection = detector.detect(request.frame_rgb, info=request.info)
                smoothed = self._smooth_detection(
                    detection=detection,
                    previous=previous,
                    info=request.info,
                    frame_shape=request.frame_rgb.shape,
                )
                with self._condition:
                    self._latest_detection = smoothed
        except BaseException as exc:
            with self._condition:
                self._worker_exception = exc
                self._condition.notify_all()
            self._ready.set()
        finally:
            if detector is not None:
                close = getattr(detector, "close", None)
                if callable(close):
                    close()

    def _smooth_detection(
        self,
        *,
        detection: BullseyeDetection | None,
        previous: BullseyeDetection | None,
        info: dict[str, object],
        frame_shape: tuple[int, ...],
    ) -> BullseyeDetection | None:
        if detection is None:
            return None
        if previous is None or self.smoothing_alpha >= 1.0:
            return detection

        width = int(info.get("width", frame_shape[1]))
        height = int(info.get("height", frame_shape[0]))
        alpha = self.smoothing_alpha
        pixel_x = alpha * detection.pixel_x + (1.0 - alpha) * previous.pixel_x
        pixel_y = alpha * detection.pixel_y + (1.0 - alpha) * previous.pixel_y
        return build_detection(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            width=width,
            height=height,
            score=detection.score,
        )

    def _raise_if_worker_failed(self) -> None:
        if self._worker_exception is not None:
            raise RuntimeError("AsyncCvBullseyeVision worker failed") from self._worker_exception
