# Reinforcement Learning Control for a Wheel-Leg Robot in Isaac Lab

Author: Luxuan Sun

This repository is for a FURP summer research project on reinforcement learning control for a wheel-leg robot in NVIDIA Isaac Lab.

The exact robot model, baseline, Isaac Lab version, training setup, and evaluation target are still to be confirmed after inspecting the initial demo or reference implementation.

## Current Status

This repository is in the initial setup stage.

Near-term goals:

1. Inspect the provided demo or baseline.
2. Document the required environment and execution model.
3. Define and run a minimal smoke test.
4. Reproduce the baseline result.
5. Compare feasible research extensions.

## Repository Structure

```text
.
├── docs/                 Public project documentation and reproducibility notes
│   ├── 00_weekly.md      Weekly progress, blockers, and next steps
│   ├── setup.md          Environment, installation, and execution notes
│   ├── experiment_log.md Reproducible experiment records and results
│   ├── meeting_notes/    Meeting summaries and action items
│   └── decisions/        Short records of important technical decisions
├── src/                  Project source code after the baseline is understood
├── scripts/              Utility scripts for setup, smoke tests, training, or evaluation
├── configs/              Robot, task, training, and experiment configuration files
└── tests/                Lightweight checks and tests that can run without full training
```

## Notes

Large generated artifacts such as checkpoints, logs, rendered outputs, and experiment runs should not be committed unless explicitly needed.
