# Setup Notes

This document records reproducible setup and environment validation notes for the project.

## Status

The project is still in the initial setup stage.

The exact robot model, baseline implementation, Isaac Lab version, Isaac Sim version, training command, and evaluation target are still to be confirmed after inspecting the received demo or reference implementation.

## Local / Remote Roles

- LOCAL: planning, documentation, Git operations, lightweight code editing, ROS 2 learning, and MuJoCo / mjlab smoke tests.
- REMOTE / GPU MACHINE: Isaac Lab execution, large-scale training, evaluation, rendering, and other GPU-heavy work.
- GITHUB: source-of-truth repository for source code and public documentation.

## Validated Local Compatibility

A local WSL2 environment with an NVIDIA laptop GPU has been validated for MuJoCo and mjlab smoke tests.

Validated capabilities:

- MuJoCo model loading, CPU stepping, and offscreen rendering.
- MuJoCo Warp GPU execution.
- mjlab runtime import and built-in task discovery.
- Small Cartpole training smoke tests.
- Small-to-moderate Unitree Go1 flat velocity smoke tests.
- A roughly 10-minute Go1 flat velocity stability smoke test at 256 parallel environments.
- Native MuJoCo viewer startup under WSLg, with unstable shutdown behavior.
- mjlab Viser browser viewer startup and timeout-triggered shutdown.

This validation does not prove:

- Isaac Lab or Isaac Sim local compatibility.
- Long-running training stability for final tasks.
- High environment-count legged robot training headroom.
- Reliable native MuJoCo / GLFW viewer shutdown under WSLg.
- Camera/depth-sensor, rough-terrain, video-recording, or large-model memory headroom.

## MuJoCo / mjlab Smoke Tests

The compatibility scripts live under `scripts/compat/`.

Basic MuJoCo compatibility:

```bash
source .venv/bin/activate
python scripts/compat/mujoco_compat_test.py
python scripts/compat/mujoco_compat_test.py --mjx
python scripts/compat/mujoco_compat_test.py --warp
```

mjlab scale smoke tests:

```bash
python scripts/compat/mjlab_scale_runner.py
```

mjlab Go1 stability smoke test:

```bash
python scripts/compat/mjlab_stability_runner.py
```

Generated outputs are written under `outputs/compat/` and should not be committed.

## To Confirm

- Isaac Lab version
- Isaac Sim version
- Python version for the final baseline
- CUDA / driver requirements
- PyTorch version for the final baseline
- Robot model and assets
- Training and evaluation commands
- Whether the demo is direct install, Conda, venv, Docker, or a prebuilt image
- Whether GUI rendering or headless execution is required
- Whether mjlab will be used as a learning aid, an alternative simulator, or part of the final project direction
