#!/usr/bin/env python3
"""CPU MuJoCo versus GPU MuJoCo Warp wheel--box contact reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


HERE = Path(__file__).resolve().parent
MODEL_PATHS = {
    "original": HERE / "models" / "robot_physical_yaw_center_root.xml",
    "reduced": HERE / "models" / "robot_collision_reduced.xml",
}
STATE_PATH = HERE / "captured_state.json"
WHEEL_NAMES = ("wheel_left_collision", "wheel_right_collision")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(name: str, path: Path) -> dict[str, Any]:
    meshes = path.parent / "meshes"
    return {
        "variant": name,
        "file": str(path.relative_to(HERE)),
        "sha256": file_sha256(path),
        "mesh_sha256": {
            str(mesh.relative_to(HERE)): file_sha256(mesh)
            for mesh in sorted(meshes.glob("*.stl"))
        }
        if name == "original"
        else {},
    }


def configure(model: mujoco.MjModel, settings: dict[str, Any]) -> None:
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.iterations = settings["iterations"]
    model.opt.ls_iterations = settings["line_search_iterations"]
    model.opt.ccd_iterations = settings["ccd_iterations"]


def add_step(model_path: Path, box: dict[str, Any], shift: np.ndarray) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(model_path))
    geom = spec.worldbody.add_geom()
    geom.name = box["name"]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = np.asarray(box["position"]) + shift
    geom.size = box["half_size"]
    geom.contype = 1
    geom.conaffinity = 1
    return spec.compile()


def wheel_contacts(data: mujoco.MjData, model: mujoco.MjModel) -> list[dict[str, Any]]:
    wheel_ids = {
        geom_id
        for name in WHEEL_NAMES
        if (geom_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
    }
    step_id = model.geom("terrain_step").id
    centers = {geom_id: data.geom_xpos[geom_id].copy() for geom_id in wheel_ids}
    result = []
    for contact in data.contact:
        pair = (int(contact.geom[0]), int(contact.geom[1]))
        if step_id not in pair or not wheel_ids.intersection(pair):
            continue
        wheel_id = next(geom_id for geom_id in pair if geom_id in wheel_ids)
        position = np.asarray(contact.pos).copy()
        result.append(
            {
                "wheel": model.geom(wheel_id).name,
                "distance": float(contact.dist),
                "position": position.tolist(),
                "distance_to_wheel_center_m": float(np.linalg.norm(position - centers[wheel_id])),
            }
        )
    return result


def run_backend_pair(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    settings: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos
    mujoco.mj_forward(model, cpu)
    cpu_contacts = wheel_contacts(cpu, model)

    seed = mujoco.MjData(model)
    seed.qpos[:] = qpos
    with wp.ScopedDevice(device):
        gpu_model = mjw.put_model(model)
        gpu_data = mjw.put_data(
            model,
            seed,
            nconmax=settings["nconmax"],
            njmax=settings["njmax"],
        )
        mjw.forward(gpu_model, gpu_data)

    raw_count = min(int(gpu_data.nacon.numpy()[0]), gpu_data.naconmax)
    raw_position = gpu_data.contact.pos.numpy()[:raw_count]
    raw_geom = gpu_data.contact.geom.numpy()[:raw_count].astype(int)
    raw_world = gpu_data.contact.worldid.numpy()[:raw_count].astype(int)
    raw_contacts = [
        {
            "position": raw_position[index].tolist(),
            "geom_ids": raw_geom[index].tolist(),
            "geom_names": [model.geom(int(geom_id)).name for geom_id in raw_geom[index]],
            "world_id": int(raw_world[index]),
        }
        for index in range(raw_count)
    ]

    gpu = mujoco.MjData(model)
    mjw.get_data_into(gpu, model, gpu_data)
    gpu_contacts = wheel_contacts(gpu, model)
    return {
        "cpu": {
            "wheel_box_contacts": cpu_contacts,
            "qacc_finite": bool(np.isfinite(cpu.qacc).all()),
        },
        "gpu": {
            "wheel_box_contacts": gpu_contacts,
            "raw_active_contact_count": raw_count,
            "raw_contacts": raw_contacts,
            "qacc_finite": bool(np.isfinite(gpu.qacc).all()),
        },
    }


def full_robot_cases(
    state: dict[str, Any], device: str, model_path: Path
) -> dict[str, Any]:
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    result = {}
    for name, shift_values in (
        ("original_world", [0.0, 0.0, 0.0]),
        ("translated_near_origin", state["translation_to_origin"]),
    ):
        shift = np.asarray(shift_values, dtype=np.float64)
        model = add_step(model_path, state["box"], shift)
        configure(model, state["solver"])
        shifted_qpos = qpos.copy()
        shifted_qpos[:3] += shift
        result[name] = run_backend_pair(model, shifted_qpos, state["solver"], device)
    return result


def cylinder_model(
    wheel_position: np.ndarray,
    wheel_rotation: np.ndarray,
    box: dict[str, Any],
    shift: np.ndarray,
) -> mujoco.MjModel:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, wheel_rotation)
    wheel_position = wheel_position + shift
    box_position = np.asarray(box["position"]) + shift
    xml = f"""
<mujoco model="single_cylinder_control">
  <option solver="Newton" cone="elliptic" iterations="50" ls_iterations="20"/>
  <default>
    <geom contype="1" conaffinity="1" condim="6"
          friction="1.7651602 0.02 0.002" solref="0.005 1"
          solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="terrain_step" type="box"
          pos="{' '.join(map(str, box_position))}"
          size="{' '.join(map(str, box['half_size']))}"/>
    <body name="wheel" pos="{' '.join(map(str, wheel_position))}"
          quat="{' '.join(map(str, quaternion))}">
      <freejoint/>
      <geom name="wheel_right_collision" type="cylinder" size="0.1 0.018"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def single_cylinder_cases(
    state: dict[str, Any], device: str, model_path: Path
) -> dict[str, Any]:
    base_model = mujoco.MjModel.from_xml_path(str(model_path))
    base_data = mujoco.MjData(base_model)
    base_data.qpos[:] = state["qpos"]
    mujoco.mj_forward(base_model, base_data)
    wheel_id = base_model.geom("wheel_right_collision").id
    wheel_position = base_data.geom_xpos[wheel_id].copy()
    wheel_rotation = base_data.geom_xmat[wheel_id].copy()

    result = {}
    for name, shift_values in (
        ("original_world", [0.0, 0.0, 0.0]),
        ("translated_near_origin", state["translation_to_origin"]),
    ):
        shift = np.asarray(shift_values, dtype=np.float64)
        model = cylinder_model(wheel_position, wheel_rotation, state["box"], shift)
        configure(model, state["solver"])
        result[name] = run_backend_pair(model, model.qpos0.copy(), state["solver"], device)
    return result


def fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    original = result["full_robot"]["original_world"]
    translated = result["full_robot"]["translated_near_origin"]
    controls = result["single_cylinder"]
    gpu_original = original["gpu"]["wheel_box_contacts"]
    max_gpu_distance = max(
        (entry["distance_to_wheel_center_m"] for entry in gpu_original),
        default=0.0,
    )
    return {
        "full_original_cpu_contact_count": len(original["cpu"]["wheel_box_contacts"]),
        "full_original_gpu_contact_count": len(gpu_original),
        "full_original_gpu_max_distance_to_wheel_m": max_gpu_distance,
        "full_original_gpu_qacc_finite": original["gpu"]["qacc_finite"],
        "full_translated_cpu_contact_count": len(translated["cpu"]["wheel_box_contacts"]),
        "full_translated_gpu_contact_count": len(translated["gpu"]["wheel_box_contacts"]),
        "control_all_qacc_finite": all(
            case[backend]["qacc_finite"]
            for case in controls.values()
            for backend in ("cpu", "gpu")
        ),
        "control_each_backend_has_contact": all(
            len(case[backend]["wheel_box_contacts"]) == 1
            for case in controls.values()
            for backend in ("cpu", "gpu")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp GPU device")
    parser.add_argument(
        "--robot-model",
        choices=tuple(MODEL_PATHS),
        default="original",
        help="use the exact original robot asset or the collision-only control",
    )
    parser.add_argument("--output", type=Path, help="write complete JSON result")
    parser.add_argument(
        "--expect-known-bug",
        action="store_true",
        help="return nonzero unless the recorded 3.8.1 failure fingerprint appears",
    )
    args = parser.parse_args()

    state = json.loads(STATE_PATH.read_text())
    model_path = MODEL_PATHS[args.robot_model]
    warp_device = wp.get_device(args.device)
    result = {
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_warp": getattr(mjw, "__version__", "unknown"),
            "warp": importlib.metadata.version("warp-lang"),
        },
        "device": {
            "alias": args.device,
            "name": warp_device.name,
            "compute_arch": getattr(warp_device, "arch", None),
        },
        "robot_model": model_metadata(args.robot_model, model_path),
        "full_robot": full_robot_cases(state, args.device, model_path),
        "single_cylinder": single_cylinder_cases(state, args.device, model_path),
    }
    result["fingerprint"] = fingerprint(result)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")

    if args.expect_known_bug:
        observed = result["fingerprint"]
        expected = (
            observed["full_original_cpu_contact_count"] > 0
            and observed["full_original_gpu_contact_count"] > 0
            and observed["full_original_gpu_max_distance_to_wheel_m"] > 1.0
            and not observed["full_original_gpu_qacc_finite"]
            and observed["full_translated_cpu_contact_count"] > 0
            and observed["full_translated_gpu_contact_count"] == 0
            and observed["control_all_qacc_finite"]
            and observed["control_each_backend_has_contact"]
        )
        if not expected:
            print("MuJoCo Warp 3.8.1 failure fingerprint was not reproduced.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
