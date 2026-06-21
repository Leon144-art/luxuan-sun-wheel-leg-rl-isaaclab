# Experiment Log

Use this file to record reproducible experiment runs and compatibility checks.

| ID | Date | Commit | Machine | Config | Seed | Command | Runtime | Result | Artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Initial placeholder |
| compat-001 | 2026-06-16 | pre-upload workspace | WSL2 GPU workstation | MuJoCo 3.9, MJX 3.9, MuJoCo Warp 3.9 | N/A | `python scripts/compat/mujoco_compat_test.py` and optional `--mjx` / `--warp` checks | <1 min after setup and cache warmup | Passed: MuJoCo stepping, offscreen render, MJX CPU execution, MuJoCo Warp GPU execution | `outputs/compat/mujoco/` | Viewer behavior tested separately in compat-005 |
| compat-002 | 2026-06-16 | pre-upload workspace | WSL2 GPU workstation | mjlab 1.4, MuJoCo 3.8, MuJoCo Warp 3.8, PyTorch CUDA runtime | 42 | Minimal `Mjlab-Cartpole-Balance` training smoke test | ~1 min including first kernel compilation | Passed: mjlab runtime import, CUDA detection, 1 PPO iteration | `outputs/compat/mjlab/rsl_rl/` | Proves minimal runtime path only |
| compat-003 | 2026-06-16 | pre-upload workspace | WSL2 GPU workstation | mjlab scale smoke tests | 42 | `python scripts/compat/mjlab_scale_runner.py` | ~2.3 min total | Passed: Cartpole 256/1024/4096 envs and Go1 flat velocity 64/128/256 envs | `outputs/compat/mjlab/scale_tests/` | Small-to-moderate smoke tests only |
| compat-004 | 2026-06-17 | pre-upload workspace | WSL2 GPU workstation | `Mjlab-Velocity-Flat-Unitree-Go1`, 256 envs | 42 | `python scripts/compat/mjlab_stability_runner.py` | ~10 min | Passed: 1200 iterations, no stderr, GPU memory released after completion | `outputs/compat/mjlab/stability_go1_256_it1200/` | Moderate sustained workload, not a maximum VRAM pressure test |
| compat-005 | 2026-06-17 | pre-upload workspace | WSL2 GPU workstation | MuJoCo native viewer and mjlab play viewers | N/A | `python scripts/compat/mujoco_compat_test.py --viewer`; `play Mjlab-Cartpole-Balance --viewer native/viser` | ~3 min total | Partial: native viewer opens but shutdown is unreliable; mjlab Viser starts and stops cleanly | `outputs/compat/mujoco/viewer_repeat/`, `outputs/compat/mjlab/viewer_*_cartpole/` | Prefer headless or Viser locally; native WSLg viewer is not a reliable path |
