from __future__ import annotations

from typer.testing import CliRunner

from target_lock.commands import app as app_module
from target_lock.commands.config import MoveCommandConfig
from target_lock.controllers import PidAimController
from target_lock.runner import AlignmentThreshold, BullseyeSource
from target_lock.runner import move as move_module


runner = CliRunner()


def test_move_command_loads_config_and_dispatches(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_move(config: MoveCommandConfig):
        captured["config"] = config
        return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(app_module, "run_move", fake_run_move)

    result = runner.invoke(app_module.app, ["move", "--max-steps", "5", "--no-fire"])

    assert result.exit_code == 0, result.stdout
    config = captured["config"]
    assert isinstance(config, MoveCommandConfig)
    assert config.session.max_steps == 5
    assert config.session.fire_when_aligned is False


def test_run_move_builds_runner_with_move_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    detector = object()

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["run_called"] = True
            return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(move_module, "Runner", FakeRunner)
    monkeypatch.setattr(move_module, "build_bullseye_detector", lambda **kwargs: detector)

    move_module.run_move(MoveCommandConfig())

    assert captured["run_called"] is True
    assert isinstance(captured["controller"], PidAimController)
    assert captured["max_steps"] is None
    assert captured["fire_when_aligned"] is True
    assert captured["bullseye_source"] == BullseyeSource.VISION
    assert captured["bullseye_detector"] is detector
    assert captured["vision_detect_every_n_frames"] == 1
    assert captured["vision_smoothing_alpha"] == 1.0
    assert callable(captured["action_mutator"])
    assert captured["threshold"] == AlignmentThreshold(
        azimuth_deg=0.18,
        elevation_deg=0.18,
        plane_x=0.01,
        plane_y=0.01,
    )
