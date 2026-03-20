# target-lock

`target-lock` is a target-locking controller demo package that integrates with the `lockon` simulator over gRPC.

## Features

- Back-projection utilities for converting image-plane observations into aiming geometry
- PID-based tracking demo for moving platform scenarios
- OmegaConf-backed move configuration with a default example YAML

## Install

```bash
pip install -e .
```

If `lockon` is not installed in the active environment, you can point to its source tree:

```powershell
$env:TARGET_LOCK_LOCKON_PATH="D:\academic\python\lockon\src"
```

## Run

Unified Typer CLI:

```bash
target-lock move
target-lock move --config examples/move/config.yaml
```

Legacy entrypoints are still supported:

```bash
target-lock-move
```

The default configuration lives at `examples/move/config.yaml`.
Tune motion, PID, alignment, vision, and runtime settings there instead of passing a long CLI argument list.

```powershell
$env:TARGET_LOCK_ONNX_PATH="D:\academic\python\autoaim\yolo\point_yolo_v8.onnx"
```

The example config reads `TARGET_LOCK_ONNX_PATH` through OmegaConf interpolation by default.

The `lockon` environment expects a 6-element action vector:

```text
[move_x, move_y, base_rot, turret_yaw, turret_pitch, fire]
```
