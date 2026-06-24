from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


KINOVA_INIT_QPOS = np.asarray([0.0, 0.650, 0.0, 1.890, 0.0, 0.600, -math.pi / 2], dtype=np.float64)


@dataclass(frozen=True)
class IKResult:
    converged: bool
    qpos: np.ndarray
    score: float
    pos_err_m: float
    rot_err_rad: float
    link_err_m: float
    active_joint_delta_norm: float
    seed_id: int
    iters: int


def require_mujoco():
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError("mujoco is required for Phantom-native explicit IK rendering.") from exc
    return mujoco


def add_phantom_submodules_to_path() -> None:
    from phantom.qwenrobot.prepare_egodex_for_phantom import PHANTOM_ROBOMIMIC, PHANTOM_ROBOSUITE

    for path in (PHANTOM_ROBOSUITE, PHANTOM_ROBOMIMIC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def export_phantom_kinova_xml(xml_path: Path) -> None:
    xml_path = Path(xml_path)
    if xml_path.exists():
        return
    add_phantom_submodules_to_path()
    from robosuite.controllers import load_controller_config
    from robomimic.envs.env_robosuite import EnvRobosuite
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=["frontview_image"]))
    )
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["control_delta"] = False
    controller_config["uncouple_pos_ori"] = False
    options: dict[str, Any] = dict(
        env_name="PhantomBimanual",
        bimanual_setup="shoulders",
        robots=["Kinova3", "Kinova3"],
        gripper_types=["Robotiq85GripperRealKinova", "Robotiq85GripperRealKinova"],
        controller_configs=controller_config,
        camera_heights=120,
        camera_widths=160,
        camera_segmentations="instance",
        direct_gripper_control=True,
        use_depth_obs=True,
        camera_pos=np.zeros(3),
        camera_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        camera_sensorsize=np.asarray([1.0, 1.0]),
        camera_principalpixel=np.asarray([0.0, 0.0]),
        camera_focalpixel=np.asarray([100.0, 100.0]),
    )
    env = EnvRobosuite(
        **options,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        camera_names=["frontview"],
        control_freq=20,
    )
    try:
        env.reset()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(env.env.sim.model.get_xml(), encoding="utf-8")
    finally:
        env.env.close()


def load_model(xml_path: Path):
    mujoco = require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    data = mujoco.MjData(model)
    return mujoco, model, data


def body_id(mujoco, model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        raise ValueError(f"Body {name!r} not found. Bodies: {names}")
    return int(bid)


def body_point(data, body: int, local_offset: np.ndarray | None = None) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float64) if local_offset is None else np.asarray(local_offset, dtype=np.float64)
    rot = data.xmat[int(body)].reshape(3, 3)
    return data.xpos[int(body)] + rot @ offset


def clamp_qpos(model, qpos: np.ndarray) -> None:
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        if adr < len(qpos):
            lo, hi = model.jnt_range[joint_id]
            qpos[adr] = np.clip(qpos[adr], lo, hi)


def set_joint_qpos(mujoco, model, data, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    adr = int(model.jnt_qposadr[joint_id])
    if adr < model.nq:
        data.qpos[adr] = float(value)


def active_dof_mask(mujoco, model, joint_prefixes: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(model.nv, dtype=bool)
    dof_counts = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not any(name.startswith(prefix) for prefix in joint_prefixes):
            continue
        dof_adr = int(model.jnt_dofadr[joint_id])
        dof_count = dof_counts.get(int(model.jnt_type[joint_id]), 1)
        mask[dof_adr : dof_adr + dof_count] = True
    if not np.any(mask):
        raise ValueError(f"No active DOFs matched joint prefixes {joint_prefixes}")
    return mask


def active_qpos_indices(mujoco, model, joint_prefixes: tuple[str, ...]) -> np.ndarray:
    indices: list[int] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not any(name.startswith(prefix) for prefix in joint_prefixes):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        if adr < model.nq:
            indices.append(adr)
    if not indices:
        raise ValueError(f"No active qpos indices matched joint prefixes {joint_prefixes}")
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def neutral_limited_qpos(mujoco, model) -> np.ndarray:
    qpos = np.zeros(model.nq, dtype=np.float64)
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            continue
        adr = int(model.jnt_qposadr[joint_id])
        if adr >= model.nq or not bool(model.jnt_limited[joint_id]):
            continue
        lo, hi = model.jnt_range[joint_id]
        qpos[adr] = 0.5 * (float(lo) + float(hi))
    return qpos


def set_initial_kinova_qpos(mujoco, model, data) -> None:
    data.qpos[:] = 0.0
    for idx, value in enumerate(KINOVA_INIT_QPOS, start=1):
        set_joint_qpos(mujoco, model, data, f"robot0_Actuator{idx}", float(value))
        set_joint_qpos(mujoco, model, data, f"robot1_Actuator{idx}", float(value))
    mujoco.mj_forward(model, data)


def set_robotiq85_width(mujoco, model, data, prefix: str, width_m: float) -> None:
    opening = float(np.clip(width_m / 0.085, 0.0, 1.0))
    driver = float(np.interp(opening, [0.0, 1.0], [0.9, 0.0]))
    for joint in (
        f"{prefix}_left_driver_joint",
        f"{prefix}_left_spring_link_joint",
        f"{prefix}_left_follower",
        f"{prefix}_right_driver_joint",
        f"{prefix}_right_spring_link_joint",
        f"{prefix}_right_follower_joint",
    ):
        set_joint_qpos(mujoco, model, data, joint, driver)


def style_kinova_robot(mujoco, model) -> None:
    hidden = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot_rgba = np.asarray([0.82, 0.84, 0.86, 1.0], dtype=np.float32)
    gripper_rgba = np.asarray([0.08, 0.08, 0.08, 1.0], dtype=np.float32)
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        is_robot = body_name.startswith("robot") or body_name.startswith("gripper")
        if not is_robot:
            model.geom_rgba[geom_id] = hidden
        elif body_name.startswith("gripper") or "finger" in body_name or "pad" in body_name:
            model.geom_rgba[geom_id] = gripper_rgba
        else:
            model.geom_rgba[geom_id] = robot_rgba
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]


def configure_camera_space_zed(mujoco, model, fovy: float) -> str:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed")
    if cam_id < 0:
        raise ValueError("Compiled Phantom XML does not contain a zed camera.")
    model.cam_pos[cam_id] = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    model.cam_quat[cam_id] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    model.cam_fovy[cam_id] = float(fovy)
    return "zed"


def render_rgb_mask_depth(mujoco, renderer, model, data, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().astype(np.float32)
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    geom_ids = seg[:, :, 0].astype(np.int32)
    mask = np.zeros(geom_ids.shape, dtype=bool)
    for geom_id in np.unique(geom_ids):
        if geom_id < 0 or geom_id >= model.ngeom:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("robot") or body_name.startswith("gripper"):
            mask |= geom_ids == geom_id
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    depth = depth.astype(np.float32)
    depth[~mask] = 0.0
    return rgb, mask, depth


def _rotation_residual(current_rot: np.ndarray, target_rot: np.ndarray | None) -> tuple[np.ndarray, float]:
    if target_rot is None:
        return np.zeros(3, dtype=np.float64), 0.0
    rotvec = R.from_matrix(np.asarray(target_rot, dtype=np.float64) @ current_rot.T).as_rotvec()
    return rotvec, float(np.linalg.norm(rotvec))


def _dedupe_seeds(seeds: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for name, seed in seeds:
        if not np.isfinite(seed).all():
            continue
        if not any(np.allclose(seed, existing, atol=1e-7, rtol=0.0) for _, existing in out):
            out.append((name, seed.copy()))
    return out


def ik_seeds(mujoco, model, previous_qpos: np.ndarray, active_prefixes: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
    prev = np.asarray(previous_qpos, dtype=np.float64).copy()
    seeds: list[tuple[str, np.ndarray]] = [("previous", prev)]
    init = prev.copy()
    for idx, value in enumerate(KINOVA_INIT_QPOS, start=1):
        for robot in ("robot0", "robot1"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_Actuator{idx}")
            if jid >= 0:
                init[int(model.jnt_qposadr[jid])] = float(value)
    seeds.append(("kinova_init", init))
    neutral = prev.copy()
    active = active_qpos_indices(mujoco, model, active_prefixes)
    neutral_all = neutral_limited_qpos(mujoco, model)
    neutral[active] = neutral_all[active]
    seeds.append(("neutral_active", neutral))
    for delta in (0.15, -0.15, 0.30, -0.30):
        seed = prev.copy()
        seed[active] += delta
        seeds.append((f"previous_{delta:+.2f}", seed))
    return _dedupe_seeds(seeds)


def solve_arm_ik_selected(
    mujoco,
    model,
    data,
    *,
    target_pos: np.ndarray,
    target_rot: np.ndarray | None,
    ee_body_id: int,
    link_targets: list[tuple[int, np.ndarray, float]],
    previous_qpos: np.ndarray,
    active_joint_prefixes: tuple[str, ...],
    pos_tol: float = 0.035,
    rot_tol_rad: float = 0.65,
    link_tol: float = 0.10,
    max_iters: int = 220,
    pos_step_limit: float = 0.30,
    damping: float = 1e-4,
) -> IKResult:
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_rot_arr = None if target_rot is None else np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    dof_mask = active_dof_mask(mujoco, model, active_joint_prefixes)
    active_qpos = active_qpos_indices(mujoco, model, active_joint_prefixes)
    best: IKResult | None = None

    for seed_id, (_seed_name, seed) in enumerate(ik_seeds(mujoco, model, previous_qpos, active_joint_prefixes)):
        data.qpos[:] = seed
        clamp_qpos(model, data.qpos)
        mujoco.mj_forward(model, data)
        local_damping = float(damping * (1.0 + seed_id))
        iters = 0
        for it in range(1, max(1, int(max_iters)) + 1):
            iters = it
            ee_pos = body_point(data, ee_body_id)
            current_rot = data.xmat[ee_body_id].reshape(3, 3)
            pos_residual = target_pos - ee_pos
            rot_residual, rot_err = _rotation_residual(current_rot, target_rot_arr)
            rows = []
            residuals = []
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(model, data, jacp, jacr, ee_pos, ee_body_id)
            rows.append(jacp)
            residuals.append(pos_residual)
            if target_rot_arr is not None:
                rows.append(0.25 * jacr)
                residuals.append(0.25 * rot_residual)
            link_errors = []
            for link_body_id, link_target, weight in link_targets:
                if weight <= 0.0:
                    continue
                link_pos = body_point(data, link_body_id)
                link_residual = np.asarray(link_target, dtype=np.float64).reshape(3) - link_pos
                link_errors.append(float(np.linalg.norm(link_residual)))
                jacp_link = np.zeros((3, model.nv), dtype=np.float64)
                jacr_link = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jac(model, data, jacp_link, jacr_link, link_pos, link_body_id)
                rows.append(float(weight) * jacp_link)
                residuals.append(float(weight) * link_residual)
            pos_err = float(np.linalg.norm(pos_residual))
            link_err = float(np.mean(link_errors)) if link_errors else 0.0
            if pos_err <= pos_tol and rot_err <= rot_tol_rad and (not link_errors or link_err <= link_tol):
                break
            jac = np.vstack(rows)
            residual = np.concatenate(residuals)
            lhs = jac @ jac.T + local_damping * np.eye(jac.shape[0])
            try:
                dq = jac.T @ np.linalg.solve(lhs, residual)
            except np.linalg.LinAlgError:
                dq = jac.T @ np.linalg.lstsq(lhs, residual, rcond=None)[0]
            dq *= dof_mask
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > pos_step_limit:
                dq *= pos_step_limit / dq_norm
            old_qpos = data.qpos.copy()
            old_score = pos_err + 0.25 * rot_err + 0.40 * link_err
            accepted = False
            for step in (1.0, 0.5, 0.25, 0.10):
                data.qpos[:] = old_qpos
                mujoco.mj_integratePos(model, data.qpos, dq, step)
                clamp_qpos(model, data.qpos)
                mujoco.mj_forward(model, data)
                new_pos = body_point(data, ee_body_id)
                new_rot = data.xmat[ee_body_id].reshape(3, 3)
                _, new_rot_err = _rotation_residual(new_rot, target_rot_arr)
                new_link_errors = [
                    float(np.linalg.norm(np.asarray(t, dtype=np.float64).reshape(3) - body_point(data, bid)))
                    for bid, t, w in link_targets
                    if w > 0.0
                ]
                new_score = (
                    float(np.linalg.norm(target_pos - new_pos))
                    + 0.25 * new_rot_err
                    + 0.40 * (float(np.mean(new_link_errors)) if new_link_errors else 0.0)
                )
                if new_score <= old_score + 1e-9:
                    accepted = True
                    local_damping = max(local_damping * 0.7, 1e-7)
                    break
            if not accepted:
                data.qpos[:] = old_qpos
                mujoco.mj_forward(model, data)
                local_damping = min(local_damping * 4.0, 1.0)
                if dq_norm < 1e-9:
                    break

        ee_pos = body_point(data, ee_body_id)
        current_rot = data.xmat[ee_body_id].reshape(3, 3)
        _, rot_err = _rotation_residual(current_rot, target_rot_arr)
        pos_err = float(np.linalg.norm(target_pos - ee_pos))
        link_errors = [
            float(np.linalg.norm(np.asarray(t, dtype=np.float64).reshape(3) - body_point(data, bid)))
            for bid, t, w in link_targets
            if w > 0.0
        ]
        link_err = float(np.mean(link_errors)) if link_errors else 0.0
        joint_delta = float(np.linalg.norm(data.qpos[active_qpos] - np.asarray(previous_qpos, dtype=np.float64)[active_qpos]))
        converged = pos_err <= pos_tol and rot_err <= rot_tol_rad and (not link_errors or link_err <= link_tol)
        score = pos_err + 0.25 * rot_err + 0.40 * link_err + 0.015 * joint_delta + (0.0 if converged else 10.0)
        result = IKResult(converged, data.qpos.copy(), score, pos_err, rot_err, link_err, joint_delta, seed_id, iters)
        if best is None or result.score < best.score:
            best = result

    if best is None:
        raise RuntimeError("IK seed generation produced no candidates")
    data.qpos[:] = best.qpos
    mujoco.mj_forward(model, data)
    return best


def ik_position(
    mujoco,
    model,
    data,
    target: np.ndarray,
    ee_body_id: int,
    *,
    initial_qpos: np.ndarray | None = None,
    active_joint_prefixes: tuple[str, ...] | None = None,
    tol: float = 0.035,
    max_iters: int = 220,
    **_: Any,
) -> tuple[bool, np.ndarray, float]:
    prefixes = active_joint_prefixes or ("robot0_Actuator", "robot1_Actuator")
    previous = data.qpos.copy() if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64)
    result = solve_arm_ik_selected(
        mujoco,
        model,
        data,
        target_pos=target,
        target_rot=None,
        ee_body_id=ee_body_id,
        link_targets=[],
        previous_qpos=previous,
        active_joint_prefixes=prefixes,
        pos_tol=tol,
        max_iters=max_iters,
    )
    return result.converged, result.qpos, result.score
