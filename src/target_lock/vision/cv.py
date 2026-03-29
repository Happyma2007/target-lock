from __future__ import annotations

import asyncio
import fractions
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import cv2
import grpc
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

from target_lock.protos.detector import common_pb2
from target_lock.protos.detector import webrtc_detector_pb2 as detector_pb2
from target_lock.protos.detector import webrtc_detector_pb2_grpc as detector_pb2_grpc
from target_lock.vision.base import BullseyeDetection, BullseyeVision, build_detection


DEFAULT_FPS = 30
MAX_PENDING_FRAMES = 120


@dataclass(slots=True)
class _QueuedFrame:
    bgr: np.ndarray


@dataclass(slots=True)
class _DetectionState:
    latest_detection: BullseyeDetection | None = None
    error: BaseException | None = None
    input_frame_id: int | None = None
    input_sync_id: int | None = None
    detection_frame_id: int | None = None
    detection_sync_id: int | None = None
    sync_status: str = "missing"


class _FrameTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        *,
        fps: int = DEFAULT_FPS,
        on_frame_sent: Callable[[int, int], None] | None = None,
    ) -> None:
        super().__init__()
        self._fps = max(int(fps), 1)
        self._time_base = fractions.Fraction(1, self._fps)
        self._start = time.time()
        self._pts = 0
        self._queue: queue.Queue[_QueuedFrame | None] = queue.Queue(maxsize=MAX_PENDING_FRAMES)
        self._on_frame_sent = on_frame_sent

    def submit_frame(self, frame_rgb: np.ndarray) -> None:
        item = _QueuedFrame(
            bgr=cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def close_track(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)

    async def recv(self) -> VideoFrame:
        self._pts += 1
        target = self._start + self._pts / self._fps
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        item = await asyncio.to_thread(self._queue.get)
        if item is None:
            raise EOFError("video track closed")

        pts_ms = self._pts_to_ms(self._pts, self._time_base)
        if self._on_frame_sent is not None:
            self._on_frame_sent(self._pts, pts_ms)

        frame = VideoFrame.from_ndarray(item.bgr, format="bgr24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        return frame

    @staticmethod
    def _pts_to_ms(pts: int, time_base: fractions.Fraction) -> int:
        return int(round(float(pts * time_base) * 1000.0))


class WebRtcBullseyeVision(BullseyeVision):
    def __init__(
        self,
        *,
        engine_addr: str,
        image_format: str = "h264",
        score_threshold: float = 0.0,
        max_detections: int = 1,
        request_timeout_s: float | None = None,
    ) -> None:
        if not engine_addr:
            raise ValueError("engine_addr must not be empty")
        if max_detections < 1:
            raise ValueError("max_detections must be >= 1")

        self.engine_addr = engine_addr
        self.video_codec = image_format.lower().strip() if image_format else "h264"
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.request_timeout_s = request_timeout_s

        self._state = _DetectionState()
        self._state_lock = threading.Lock()
        self._frame_shape: tuple[int, int] | None = None
        self._stream_started = False
        self._closed = False
        self._start_lock = threading.Lock()
        self._track = _FrameTrack(on_frame_sent=self._on_frame_sent)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="target-lock-webrtc",
            daemon=True,
        )
        self._thread.start()

    def detect(
        self,
        frame_rgb: np.ndarray,
        info: Mapping[str, object] | None = None,
    ) -> BullseyeDetection | None:
        del info
        self._raise_if_failed()
        height, width = frame_rgb.shape[:2]
        self._ensure_started(width=width, height=height)
        self._track.submit_frame(frame_rgb)
        with self._state_lock:
            return self._state.latest_detection

    def get_latest_detection(self) -> BullseyeDetection | None:
        self._raise_if_failed()
        with self._state_lock:
            return self._state.latest_detection

    def get_debug_state(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "vision_input_frame_id": self._state.input_frame_id,
                "vision_input_sync_id": self._state.input_sync_id,
                "vision_detection_frame_id": self._state.detection_frame_id,
                "vision_detection_sync_id": self._state.detection_sync_id,
                "vision_detection_status": self._state.sync_status,
            }

    def reset(self) -> None:
        with self._state_lock:
            self._state.latest_detection = None
            self._state.detection_frame_id = None
            self._state.detection_sync_id = None
            self._state.sync_status = "missing"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._track.close_track()
        future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def _ensure_started(self, *, width: int, height: int) -> None:
        with self._start_lock:
            if self._stream_started:
                return
            self._frame_shape = (width, height)
            future = asyncio.run_coroutine_threadsafe(
                self._start_session(width=width, height=height),
                self._loop,
            )
            future.result(timeout=15.0)
            self._stream_started = True

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _on_frame_sent(self, frame_id: int, sync_id: int) -> None:
        with self._state_lock:
            self._state.input_frame_id = int(frame_id)
            self._state.input_sync_id = int(sync_id)
            if self._state.detection_frame_id is None:
                self._state.sync_status = "missing"
            elif self._state.detection_frame_id == self._state.input_frame_id:
                self._state.sync_status = "synced"
            else:
                self._state.sync_status = "fallback"

    async def _start_session(self, *, width: int, height: int) -> None:
        self._channel = grpc.aio.insecure_channel(self.engine_addr)
        self._stub = detector_pb2_grpc.WebRtcDetectorEngineStub(self._channel)

        create_reply = await self._stub.CreateStream(
            detector_pb2.CreateStreamRequest(
                config=common_pb2.StreamConfig(
                    video_codec=self.video_codec,
                    width=width,
                    height=height,
                    score_threshold=self.score_threshold,
                    max_detections=self.max_detections,
                )
            ),
            timeout=self.request_timeout_s,
        )
        self._stream_id = create_reply.stream_id

        self._pc = RTCPeerConnection()
        self._pc.addTrack(self._track)

        await self._pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=create_reply.offer.sdp,
                type=create_reply.offer.type,
            )
        )
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        while self._pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        await self._stub.UpdateStream(
            detector_pb2.StreamSignal(
                stream_id=self._stream_id,
                answer=detector_pb2.SessionDescription(
                    type=self._pc.localDescription.type,
                    sdp=self._pc.localDescription.sdp,
                ),
            ),
            timeout=self.request_timeout_s,
        )
        self._recv_task = asyncio.create_task(self._receive_detections())

    async def _receive_detections(self) -> None:
        try:
            async for reply in self._stub.StreamDetections(
                detector_pb2.StreamDetectionsRequest(stream_id=self._stream_id)
            ):
                detection = self._select_detection(reply.detections)
                with self._state_lock:
                    self._state.latest_detection = detection
                    self._state.detection_frame_id = int(reply.frame_id)
                    self._state.detection_sync_id = int(reply.pts_ms)
                    if detection is None:
                        self._state.sync_status = "missing"
                    elif self._state.input_frame_id == self._state.detection_frame_id:
                        self._state.sync_status = "synced"
                    else:
                        self._state.sync_status = "fallback"
        except BaseException as exc:
            with self._state_lock:
                self._state.error = exc

    async def _shutdown_async(self) -> None:
        recv_task = getattr(self, "_recv_task", None)
        if recv_task is not None:
            recv_task.cancel()
            try:
                await recv_task
            except BaseException:
                pass
        pc = getattr(self, "_pc", None)
        if pc is not None:
            await pc.close()
        channel = getattr(self, "_channel", None)
        if channel is not None:
            await channel.close()

    def _raise_if_failed(self) -> None:
        with self._state_lock:
            if self._state.error is not None:
                raise RuntimeError("WebRtcBullseyeVision failed") from self._state.error

    def _select_detection(
        self,
        detections: list[common_pb2.Detection],
    ) -> BullseyeDetection | None:
        if not detections:
            return None
        detection = max(detections, key=lambda item: float(item.score))
        width, height = self._frame_shape or (1, 1)
        return self._to_bullseye_detection(detection, width=width, height=height)

    def _to_bullseye_detection(
        self,
        detection: common_pb2.Detection,
        *,
        width: int,
        height: int,
    ) -> BullseyeDetection:
        pixel_point = self._point_from_geometry(detection.geometry)
        normalized_point = self._point_from_geometry(detection.normalized_geometry)
        if pixel_point is not None:
            pixel_x, pixel_y = pixel_point
        elif normalized_point is not None:
            pixel_x = normalized_point[0] * float(width)
            pixel_y = normalized_point[1] * float(height)
        else:
            raise ValueError("Detector reply does not contain point geometry")

        built = build_detection(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            width=width,
            height=height,
            score=float(detection.score),
        )
        if normalized_point is None:
            return built
        return BullseyeDetection(
            pixel_x=built.pixel_x,
            pixel_y=built.pixel_y,
            score=built.score,
            x_norm=float(normalized_point[0]),
            y_norm=float(normalized_point[1]),
        )

    @staticmethod
    def _point_from_geometry(geometry: common_pb2.DetectionGeometry) -> tuple[float, float] | None:
        kind = geometry.WhichOneof("shape")
        if kind == "point":
            return float(geometry.point.x), float(geometry.point.y)
        if kind == "circle":
            return float(geometry.circle.center.x), float(geometry.circle.center.y)
        if kind == "box":
            return (
                (float(geometry.box.x_min) + float(geometry.box.x_max)) * 0.5,
                (float(geometry.box.y_min) + float(geometry.box.y_max)) * 0.5,
            )
        return None


GrpcBullseyeVision = WebRtcBullseyeVision
CvBullseyeVision = WebRtcBullseyeVision
