from __future__ import annotations

from pathlib import Path
import time
from threading import Event

import pytest
import numpy as np

from target_lock.vision import AsyncCvBullseyeVision, DEFAULT_AUTOAIM_MODEL, OracleBullseyeVision, resolve_autoaim_onnx_path
from target_lock.vision.base import BullseyeDetection


def _create_model_file(repo_dir: Path) -> Path:
    model_path = repo_dir / DEFAULT_AUTOAIM_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")
    return model_path


def test_resolve_autoaim_onnx_path_prefers_explicit_path(tmp_path: Path) -> None:
    explicit_model = tmp_path / "custom" / "model.onnx"
    explicit_model.parent.mkdir(parents=True, exist_ok=True)
    explicit_model.write_bytes(b"onnx")

    resolved = resolve_autoaim_onnx_path(None, explicit_model)

    assert resolved == explicit_model


def test_resolve_autoaim_onnx_path_requires_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARGET_LOCK_AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)

    with pytest.raises(ValueError, match="Autoaim model location is not configured"):
        resolve_autoaim_onnx_path(None, None)


def test_resolve_autoaim_onnx_path_reads_repo_from_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "env-autoaim"
    expected_model = _create_model_file(repo_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_LOCK_AUTOAIM_REPO", str(repo_dir))
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model


def test_resolve_autoaim_onnx_path_reads_repo_from_dotenv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "dotenv-autoaim"
    expected_model = _create_model_file(repo_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARGET_LOCK_AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)
    (tmp_path / ".env").write_text(f"TARGET_LOCK_AUTOAIM_REPO={repo_dir}\n", encoding="utf-8")

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model


def test_resolve_autoaim_onnx_path_prefers_environment_over_dotenv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_repo_dir = tmp_path / "env-autoaim"
    dotenv_repo_dir = tmp_path / "dotenv-autoaim"
    expected_model = _create_model_file(env_repo_dir)
    _create_model_file(dotenv_repo_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_LOCK_AUTOAIM_REPO", str(env_repo_dir))
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)
    (tmp_path / ".env").write_text(f"TARGET_LOCK_AUTOAIM_REPO={dotenv_repo_dir}\n", encoding="utf-8")

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model


def test_oracle_bullseye_vision_reads_detection_from_info() -> None:
    detector = OracleBullseyeVision()

    detection = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8),
        info={
            "bullseye_pixel": [320, 120],
            "width": 640,
            "height": 480,
        },
    )

    assert detection is not None
    assert detection.to_pixel_list() == [320.0, 120.0]
    assert detection.score == 1.0
    assert detection.x_norm == 0.5
    assert detection.y_norm == 0.25


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not satisfied before timeout")


def test_async_cv_bullseye_vision_keeps_only_latest_pending_frame() -> None:
    releases = [Event(), Event()]
    started = [Event(), Event()]
    processed_frames: list[int] = []

    class FakeDetector:
        def detect(self, frame_rgb: np.ndarray, info=None) -> BullseyeDetection | None:
            del info
            frame_id = int(frame_rgb[0, 0, 0])
            call_index = len(processed_frames)
            processed_frames.append(frame_id)
            started[call_index].set()
            releases[call_index].wait(timeout=1.0)
            return BullseyeDetection(
                pixel_x=float(frame_id),
                pixel_y=1.0,
                score=0.9,
                x_norm=0.1,
                y_norm=0.2,
            )

    detector = AsyncCvBullseyeVision(
        onnx_path="unused.onnx",
        detector_factory=FakeDetector,
    )
    try:
        frame_1 = np.full((4, 4, 3), 1, dtype=np.uint8)
        frame_2 = np.full((4, 4, 3), 2, dtype=np.uint8)
        frame_3 = np.full((4, 4, 3), 3, dtype=np.uint8)

        started_at = time.perf_counter()
        assert detector.detect(frame_1) is None
        elapsed = time.perf_counter() - started_at
        assert elapsed < 0.05

        started[0].wait(timeout=1.0)
        detector.detect(frame_2)
        detector.detect(frame_3)

        releases[0].set()
        _wait_until(lambda: started[1].is_set())
        assert processed_frames == [1, 3]

        releases[1].set()
        _wait_until(
            lambda: (
                detector.get_latest_detection() is not None
                and detector.get_latest_detection().pixel_x == 3.0
            )
        )
        latest = detector.get_latest_detection()
        assert latest is not None
        assert latest.to_pixel_list() == [3.0, 1.0]
    finally:
        detector.close()


def test_async_cv_bullseye_vision_smooths_only_on_new_results() -> None:
    releases = [Event(), Event()]
    detections = [
        BullseyeDetection(pixel_x=10.0, pixel_y=2.0, score=0.9, x_norm=0.2, y_norm=0.3),
        BullseyeDetection(pixel_x=18.0, pixel_y=2.0, score=0.9, x_norm=0.2, y_norm=0.3),
    ]

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame_rgb: np.ndarray, info=None) -> BullseyeDetection | None:
            del frame_rgb, info
            call_index = self.calls
            self.calls += 1
            releases[call_index].wait(timeout=1.0)
            return detections[call_index]

    detector = AsyncCvBullseyeVision(
        onnx_path="unused.onnx",
        smoothing_alpha=0.25,
        detector_factory=FakeDetector,
    )
    try:
        frame = np.zeros((8, 40, 3), dtype=np.uint8)
        info = {"width": 40, "height": 8}

        assert detector.detect(frame, info=info) is None
        releases[0].set()
        _wait_until(lambda: detector.get_latest_detection() is not None)

        first = detector.get_latest_detection()
        assert first is not None
        assert first.to_pixel_list() == [10.0, 2.0]
        assert detector.get_latest_detection() is first

        detector.detect(frame, info=info)
        releases[1].set()
        _wait_until(lambda: detector.get_latest_detection() is not first)

        second = detector.get_latest_detection()
        assert second is not None
        assert second.to_pixel_list() == [12.0, 2.0]
        assert detector.get_latest_detection() is second
        assert detector.get_latest_detection().to_pixel_list() == [12.0, 2.0]
    finally:
        detector.close()


def test_async_cv_bullseye_vision_clears_latest_detection_on_miss() -> None:
    releases = [Event(), Event()]

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame_rgb: np.ndarray, info=None) -> BullseyeDetection | None:
            del frame_rgb, info
            call_index = self.calls
            self.calls += 1
            releases[call_index].wait(timeout=1.0)
            if call_index == 0:
                return BullseyeDetection(pixel_x=6.0, pixel_y=3.0, score=0.9, x_norm=0.2, y_norm=0.3)
            return None

    detector = AsyncCvBullseyeVision(
        onnx_path="unused.onnx",
        detector_factory=FakeDetector,
    )
    try:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        assert detector.detect(frame) is None
        releases[0].set()
        _wait_until(lambda: detector.get_latest_detection() is not None)

        latest = detector.get_latest_detection()
        assert latest is not None
        assert latest.to_pixel_list() == [6.0, 3.0]

        detector.detect(frame)
        releases[1].set()
        _wait_until(lambda: detector.get_latest_detection() is None)
        assert detector.get_latest_detection() is None
    finally:
        detector.close()
