#!/usr/bin/env python3
"""MuJoCo compatibility smoke test.

The script intentionally uses an embedded MJCF model so it does not depend on
external robot assets or a received demo package.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


MODEL_XML = """
<mujoco model="wheel_leg_smoke">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".15 .17 .19" rgb2=".22 .24 .27"/>
    <material name="grid" texture="grid" texrepeat="2 2" reflectance="0.1"/>
    <material name="body_mat" rgba="0.1 0.35 0.75 1"/>
    <material name="leg_mat" rgba="0.9 0.55 0.08 1"/>
    <material name="wheel_mat" rgba="0.05 0.05 0.05 1"/>
  </asset>

  <worldbody>
    <light pos="0 -3 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 .05" material="grid"/>

    <body name="base" pos="0 0 0.45">
      <freejoint/>
      <geom name="base_box" type="box" size="0.22 0.12 0.06" material="body_mat" mass="2.0"/>

      <body name="left_leg" pos="0 0.13 -0.08">
        <joint name="left_hip" type="hinge" axis="0 1 0" limited="true" range="-45 45" damping="0.1"/>
        <geom name="left_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.025" material="leg_mat" mass="0.4"/>
        <body name="left_wheel" pos="0 0 -0.3">
          <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.02"/>
          <geom name="left_wheel_geom" type="cylinder" size="0.08 0.035" material="wheel_mat" mass="0.25"/>
        </body>
      </body>

      <body name="right_leg" pos="0 -0.13 -0.08">
        <joint name="right_hip" type="hinge" axis="0 1 0" limited="true" range="-45 45" damping="0.1"/>
        <geom name="right_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.025" material="leg_mat" mass="0.4"/>
        <body name="right_wheel" pos="0 0 -0.3">
          <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.02"/>
          <geom name="right_wheel_geom" type="cylinder" size="0.08 0.035" material="wheel_mat" mass="0.25"/>
        </body>
      </body>
    </body>

    <camera name="overview" pos="1.1 -1.8 1.0" xyaxes="0.85 0.53 0 -0.25 0.4 0.88"/>
  </worldbody>

  <actuator>
    <motor name="left_hip_motor" joint="left_hip" gear="5"/>
    <motor name="right_hip_motor" joint="right_hip" gear="5"/>
    <motor name="left_wheel_motor" joint="left_wheel_joint" gear="0.4"/>
    <motor name="right_wheel_motor" joint="right_wheel_joint" gear="0.4"/>
  </actuator>
</mujoco>
"""

ACCELERATOR_XML = """
<mujoco model="accelerator_smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pendulum" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="0.2"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


def run_command(args: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, "command timed out"

    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def collect_system_info() -> None:
    print_section("System")
    print_kv("python", sys.version.replace("\n", " "))
    print_kv("platform", platform.platform())
    print_kv("mujoco", mujoco.__version__)
    print_kv("numpy", np.__version__)
    print_kv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "<unset>"))
    print_kv("DISPLAY", os.environ.get("DISPLAY", "<unset>"))
    print_kv("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", "<unset>"))
    print_kv("GALLIUM_DRIVER", os.environ.get("GALLIUM_DRIVER", "<unset>"))
    print_kv("MESA_D3D12_DEFAULT_ADAPTER_NAME", os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME", "<unset>"))

    code, output = run_command(["nvidia-smi"], timeout=5)
    first_line = output.splitlines()[0] if output else "<no output>"
    print_kv("nvidia-smi", f"exit={code}; {first_line}")

    code, output = run_command(["glxinfo", "-B"], timeout=5)
    renderer = "<not found>"
    for line in output.splitlines():
        if "OpenGL renderer string" in line:
            renderer = line.split(":", 1)[1].strip()
            break
    print_kv("glxinfo renderer", f"exit={code}; {renderer}")


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    return model, data


def run_physics(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    print_section("CPU Physics")
    start = time.perf_counter()
    for i in range(steps):
        data.ctrl[0] = 0.3 * np.sin(i * 0.015)
        data.ctrl[1] = -0.3 * np.sin(i * 0.015)
        data.ctrl[2] = 0.8
        data.ctrl[3] = 0.8
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start

    print_kv("steps", steps)
    print_kv("sim_time_s", f"{data.time:.3f}")
    print_kv("wall_time_s", f"{elapsed:.3f}")
    print_kv("steps_per_s", f"{steps / elapsed:.0f}")
    print_kv("base_pos_xyz", np.array2string(data.qpos[:3], precision=4))
    print_kv("qpos_size", model.nq)
    print_kv("qvel_size", model.nv)
    print_kv("actuator_count", model.nu)


def render_offscreen(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    print_section("Offscreen Render")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="overview")
    pixels = renderer.render()
    renderer.close()

    Image.fromarray(pixels).save(output_path)
    print_kv("render_path", output_path)
    print_kv("image_shape", pixels.shape)
    print_kv("pixel_min", int(pixels.min()))
    print_kv("pixel_max", int(pixels.max()))
    print_kv("pixel_mean", f"{float(pixels.mean()):.2f}")


def probe_viewer(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    print_section("Interactive Viewer")
    try:
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - diagnostic path
        print_kv("viewer_import", f"failed: {exc!r}")
        return

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            deadline = time.time() + seconds
            frames = 0
            while viewer.is_running() and time.time() < deadline:
                mujoco.mj_step(model, data)
                viewer.sync()
                frames += 1
                time.sleep(0.01)
        print_kv("viewer", f"opened and synced {frames} frames")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print_kv("viewer", f"failed: {exc!r}")


def probe_mjx(steps: int) -> None:
    print_section("MJX / JAX")
    try:
        import jax
        import mujoco.mjx as mjx
    except Exception as exc:  # pragma: no cover - optional dependency path
        print_kv("mjx", f"unavailable: {exc!r}")
        return

    model = mujoco.MjModel.from_xml_string(ACCELERATOR_XML)
    data = mujoco.MjData(model)
    start = time.perf_counter()
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    for _ in range(steps):
        mjx_data = mjx_data.replace(ctrl=mjx_data.ctrl.at[0].set(0.1))
        mjx_data = mjx.step(mjx_model, mjx_data)
    elapsed = time.perf_counter() - start

    print_kv("jax_devices", jax.devices())
    print_kv("steps", steps)
    print_kv("wall_time_s", f"{elapsed:.3f}")
    print_kv("sim_time_s", f"{float(mjx_data.time):.3f}")
    print_kv("qpos", mjx_data.qpos)


def probe_warp(steps: int) -> None:
    print_section("MuJoCo Warp")
    try:
        import mujoco_warp as mjw
        import warp as wp
    except Exception as exc:  # pragma: no cover - optional dependency path
        print_kv("mujoco_warp", f"unavailable: {exc!r}")
        return

    wp.init()
    model = mujoco.MjModel.from_xml_string(ACCELERATOR_XML)
    data = mujoco.MjData(model)
    start = time.perf_counter()
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(model, data)
    for _ in range(steps):
        mjw.step(warp_model, warp_data)
    result = mujoco.MjData(model)
    mjw.get_data_into(result, model, warp_data)
    elapsed = time.perf_counter() - start

    print_kv("warp_devices", wp.get_devices())
    print_kv("preferred_device", wp.get_preferred_device())
    print_kv("steps", steps)
    print_kv("wall_time_s", f"{elapsed:.3f}")
    print_kv("sim_time_s", f"{result.time:.3f}")
    print_kv("qpos", np.array2string(result.qpos, precision=4))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--accelerator-steps", type=int, default=100)
    parser.add_argument("--render-path", default="outputs/compat/mujoco/smoke.png")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo interactive viewer briefly.")
    parser.add_argument("--viewer-seconds", type=float, default=3.0)
    parser.add_argument("--mjx", action="store_true", help="Probe MJX/JAX support with a simple model.")
    parser.add_argument("--warp", action="store_true", help="Probe MuJoCo Warp support with a simple model.")
    args = parser.parse_args()

    collect_system_info()
    model, data = build_model()
    run_physics(model, data, args.steps)
    render_offscreen(model, data, Path(args.render_path))

    if args.viewer:
        probe_viewer(model, data, args.viewer_seconds)
    if args.mjx:
        probe_mjx(args.accelerator_steps)
    if args.warp:
        probe_warp(args.accelerator_steps)

    print_section("Result")
    print("PASS: MuJoCo import, model load, CPU stepping, and offscreen rendering completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
