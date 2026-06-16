#!/usr/bin/env python3
"""Run a longer local mjlab stability smoke test.

The default target is a roughly 10-minute Go1 flat velocity run on a
CUDA-capable NVIDIA GPU. It is still a compatibility/stability check, not a
meaningful training run.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from threading import Event, Thread


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / ".venv-mjlab" / "bin" / "train"
OUT_DIR = ROOT / "outputs" / "compat" / "mjlab" / "stability_go1_256_it1200"
TEST_ID = "go1-flat-env256-it1200"


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def query_gpu() -> list[str] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return [part.strip() for part in result.stdout.strip().split(",", 4)]


def monitor_gpu(stop: Event, log_path: Path, interval_s: float = 1.0) -> None:
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "memory_used_mib",
                "utilization_gpu_percent",
                "power_draw_w",
                "temperature_gpu_c",
            ]
        )
        while not stop.is_set():
            row = query_gpu()
            if row is not None:
                writer.writerow(row)
                f.flush()
            time.sleep(interval_s)


def summarize_gpu(log_path: Path) -> dict[str, float | int | None]:
    values: dict[str, list[float]] = {
        "memory_used_mib": [],
        "utilization_gpu_percent": [],
        "power_draw_w": [],
        "temperature_gpu_c": [],
    }
    with log_path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in values:
                try:
                    values[key].append(float(row[key]))
                except (KeyError, ValueError):
                    pass

    summary: dict[str, float | int | None] = {"samples": len(next(iter(values.values()), []))}
    for key, series in values.items():
        summary[f"{key}_max"] = round(max(series), 3) if series else None
        summary[f"{key}_avg"] = round(sum(series) / len(series), 3) if series else None
    return summary


def extract_training_summary(stdout_path: Path) -> dict[str, str | None]:
    text = stdout_path.read_text(errors="replace")
    fields = {
        "last_total_steps": r"Total steps:\s*([0-9]+)",
        "last_steps_per_second": r"Steps per second:\s*([0-9.]+)",
        "last_iteration_time_s": r"Iteration time:\s*([0-9.]+)s",
        "time_elapsed": r"Time elapsed:\s*([0-9:]+)",
    }
    result: dict[str, str | None] = {}
    for key, pattern in fields.items():
        matches = re.findall(pattern, text)
        result[key] = matches[-1] if matches else None
    return result


def main() -> int:
    if not TRAIN.exists():
        raise SystemExit(f"Missing mjlab train executable: {TRAIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = OUT_DIR / "stdout.log"
    stderr_path = OUT_DIR / "stderr.log"
    gpu_path = OUT_DIR / "gpu.csv"
    log_root = OUT_DIR / "rsl_rl"

    cmd = [
        str(TRAIN),
        "Mjlab-Velocity-Flat-Unitree-Go1",
        "--env.scene.num-envs",
        "256",
        "--agent.max-iterations",
        "1200",
        "--agent.num-steps-per-env",
        "8",
        "--agent.logger",
        "tensorboard",
        "--agent.save-interval",
        "1200",
        "--agent.run-name",
        TEST_ID,
        "--log-root",
        str(log_root),
    ]

    print(f"Starting {TEST_ID}")
    print("Command:", " ".join(cmd))
    print(f"Output: {OUT_DIR.relative_to(ROOT)}")

    stop = Event()
    monitor = Thread(target=monitor_gpu, args=(stop, gpu_path), daemon=True)
    monitor.start()

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=base_env(),
        stdout=stdout_path.open("w"),
        stderr=stderr_path.open("w"),
        text=True,
    )

    try:
        while proc.poll() is None:
            elapsed = time.perf_counter() - start
            row = query_gpu()
            if row is not None:
                print(
                    f"elapsed={elapsed:6.1f}s mem={row[1]}MiB util={row[2]}% "
                    f"power={row[3]}W temp={row[4]}C",
                    flush=True,
                )
            time.sleep(30)
    finally:
        stop.set()
        monitor.join(timeout=2)

    elapsed = time.perf_counter() - start
    result = {
        "test_id": TEST_ID,
        "task": "Mjlab-Velocity-Flat-Unitree-Go1",
        "num_envs": 256,
        "max_iterations": 1200,
        "steps_per_env": 8,
        "returncode": proc.returncode,
        "wall_time_s": round(elapsed, 3),
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "gpu_log": str(gpu_path.relative_to(ROOT)),
        "log_root": str(log_root.relative_to(ROOT)),
        "command": " ".join(cmd),
        "gpu_summary": summarize_gpu(gpu_path),
        "training_summary": extract_training_summary(stdout_path),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
