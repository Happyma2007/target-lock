from __future__ import annotations

import os
from pathlib import Path


DEFAULT_AUTOAIM_MODEL = "yolo/point_yolo_v8.onnx"


def resolve_autoaim_onnx_path(
    autoaim_repo: str | Path | None,
    onnx_path: str | Path | None,
) -> Path:
    if onnx_path is not None:
        return Path(onnx_path).expanduser()

    dotenv_values = _read_dotenv_values(Path.cwd() / ".env")
    env_onnx_path = _get_first_defined("TARGET_LOCK_ONNX_PATH", "ONNX_PATH", dotenv_values=dotenv_values)
    if env_onnx_path is not None:
        return Path(env_onnx_path).expanduser()

    repo_root = autoaim_repo
    if repo_root is None:
        repo_root = _get_first_defined("TARGET_LOCK_AUTOAIM_REPO", "AUTOAIM_REPO", dotenv_values=dotenv_values)
    if repo_root is None:
        raise ValueError("Autoaim model location is not configured")

    return Path(repo_root).expanduser() / DEFAULT_AUTOAIM_MODEL


def _get_first_defined(*keys: str, dotenv_values: dict[str, str]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    for key in keys:
        value = dotenv_values.get(key)
        if value:
            return value
    return None


def _read_dotenv_values(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values
