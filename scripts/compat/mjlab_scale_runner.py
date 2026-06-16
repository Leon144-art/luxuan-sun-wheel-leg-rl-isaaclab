#!/usr/bin/env python3
"""Run local mjlab scale compatibility tests.

This is a bounded compatibility runner, not a training benchmark. It runs a
small number of iterations for selected tasks and records stdout, stderr, wall
time, and sampled GPU memory usage under outputs/compat/mjlab/scale_tests/.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Event, Thread


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / ".venv-mjlab" / "bin" / "train"
OUT_DIR = ROOT / "outputs" / "compat" / "mjlab" / "scale_tests"


@dataclass(frozen=True)
class TestCase:
    test_id: str
    task: str
    num_envs: int
    max_iterations: int
    steps_per_env: int = 8


TESTS = [
    TestCase("cartpole-env256-it5", "Mjlab-Cartpole-Balance", 256, 5),
    TestCase("cartpole-env1024-it5", "Mjlab-Cartpole-Balance", 1024, 5),
    TestCase("cartpole-env4096-it5", "Mjlab-Cartpole-Balance", 4096, 5),
    TestCase("go1-flat-env64-it1", "Mjlab-Velocity-Flat-Unitree-Go1", 64, 1),
    TestCase("go1-flat-env128-it1", "Mjlab-Velocity-Flat-Unitree-Go1", 128, 1),
    TestCase("go1-flat-env256-it1", "Mjlab-Velocity-Flat-Unitree-Go1", 256, 1),
]


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def monitor_gpu(stop: Event, log_path: Path, interval_s: float = 0.5) -> None:
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "memory_used_mib", "utilization_gpu_percent"])
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                row = [part.strip() for part in result.stdout.strip().split(",", 2)]
                writer.writerow(row)
                f.flush()
            time.sleep(interval_s)


def max_gpu_memory(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    max_mem: int | None = None
    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mem = int(row["memory_used_mib"])
            except (KeyError, ValueError):
                continue
            max_mem = mem if max_mem is None else max(max_mem, mem)
    return max_mem


def run_case(case: TestCase) -> dict[str, object]:
    case_dir = OUT_DIR / case.test_id
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    gpu_path = case_dir / "gpu.csv"

    log_root = OUT_DIR / "rsl_rl" / case.test_id
    cmd = [
        str(TRAIN),
        case.task,
        "--env.scene.num-envs",
        str(case.num_envs),
        "--agent.max-iterations",
        str(case.max_iterations),
        "--agent.num-steps-per-env",
        str(case.steps_per_env),
        "--agent.logger",
        "tensorboard",
        "--agent.save-interval",
        str(max(case.max_iterations, 1)),
        "--agent.run-name",
        case.test_id,
        "--log-root",
        str(log_root),
    ]

    stop = Event()
    monitor = Thread(target=monitor_gpu, args=(stop, gpu_path), daemon=True)
    monitor.start()

    start = time.perf_counter()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=base_env(),
            check=False,
            stdout=out,
            stderr=err,
            text=True,
        )
    elapsed = time.perf_counter() - start
    stop.set()
    monitor.join(timeout=2)

    result = {
        **asdict(case),
        "returncode": proc.returncode,
        "wall_time_s": round(elapsed, 3),
        "max_gpu_memory_mib": max_gpu_memory(gpu_path),
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "gpu_log": str(gpu_path.relative_to(ROOT)),
        "log_root": str(log_root.relative_to(ROOT)),
        "command": " ".join(cmd),
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    if not TRAIN.exists():
        raise SystemExit(f"Missing mjlab train executable: {TRAIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in TESTS:
        print(f"\n=== {case.test_id} ===", flush=True)
        result = run_case(case)
        results.append(result)
        print(
            "returncode={returncode} wall_time_s={wall_time_s} "
            "max_gpu_memory_mib={max_gpu_memory_mib}".format(**result),
            flush=True,
        )
        if result["returncode"] != 0:
            print(f"FAILED: see {result['stdout']} and {result['stderr']}", flush=True)
            break

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
