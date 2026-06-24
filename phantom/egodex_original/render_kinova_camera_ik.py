from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from phantom.qwenrobot.mujoco_utils import (
    active_dof_mask,
    body_id,
    body_point,
    clamp_qpos,
    set_joint_qpos,
)
from phantom.qwenrobot.render_phantom_fixed_panda import (
    fixed_root_from_forearm_border,
    fovy_from_intrinsic,
    overlay_robot,
    read_video_frames,
)
from phantom.qwenrobot.common import save_video, write_json
from phantom.qwenrobot.prepare_egodex_for_phantom import (
    PHANTOM_ROBOMIMIC,
    PHANTOM_ROBOSUITE,
    REPO_ROOT,
    load_extrinsics_matrix,
)


KINOVA_INIT_QPOS = np.asarray([0.0, 0.650, 0.0, 1.890, 0.0, 0.600, -math.pi / 2], dtype=np.float64)


def add_phantom_submodules_to_path() -> None:
    for path in (PHANTOM_ROBOSUITE, PHANTOM_ROBOMIMIC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def export_phantom_kinova_xml(xml_path: Path) -> None:
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
    options = dict(
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


def read_intrinsic(processed_demo_dir: Path) -> np.ndarray:
    candidates = [
        processed_demo_dir / "egodex_camera_intrinsics.json",
        processed_demo_dir.parents[2] / "egodex_camera_intrinsics.json",
        processed_demo_dir.parents[2] / "raw" / processed_demo_dir.parents[1].name / processed_demo_dir.name / "egodex_camera_intrinsics.json",
    ]
    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_dir = Path(str(manifest["raw_demo_dir"]))
        candidates.insert(0, raw_dir / "egodex_camera_intrinsics.json")
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))["left"]
            return np.asarray(
                [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
    raise FileNotFoundError(f"No EgoDex intrinsics found near {processed_demo_dir}")


def camera_extrinsics_path(processed_demo_dir: Path) -> Path:
    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(str(manifest["camera_extrinsics"]))
        if path.exists():
            return path
    return REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json"


def actions_to_camera(points_action: np.ndarray, extrinsics_path: Path) -> np.ndarray:
    t_cam_to_action = load_extrinsics_matrix(extrinsics_path)
    t_action_to_cam = np.linalg.inv(t_cam_to_action)
    pts_h = np.concatenate([points_action, np.ones((len(points_action), 1), dtype=points_action.dtype)], axis=1)
    return (pts_h @ t_action_to_cam.T)[:, :3].astype(np.float64)


def build_traj_like(processed_demo_dir: Path, setup: str, intrinsic: np.ndarray) -> dict[str, np.ndarray | str]:
    smooth_dir = processed_demo_dir / "smoothing_processor"
    left = np.load(smooth_dir / f"smoothed_actions_left_{setup}.npz")
    right = np.load(smooth_dir / f"smoothed_actions_right_{setup}.npz")
    extrinsics = camera_extrinsics_path(processed_demo_dir)
    manifest = json.loads((processed_demo_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    return {
        "left_ee_pos_camera": actions_to_camera(left["ee_pts"].astype(np.float64), extrinsics),
        "right_ee_pos_camera": actions_to_camera(right["ee_pts"].astype(np.float64), extrinsics),
        "left_width": left["ee_widths"].astype(np.float64),
        "right_width": right["ee_widths"].astype(np.float64),
        "camera_intrinsic": intrinsic,
        "source_hdf5": str(manifest["source_hdf5"]),
        "frame_indices": np.asarray(manifest["selected_source_indices"], dtype=np.int64),
    }


def world_points_to_camera(points_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    camera_pos = camera_transforms[:, :3, 3]
    camera_rot = camera_transforms[:, :3, :3]
    camera_rot_t = np.swapaxes(camera_rot, 1, 2)
    return np.einsum("nij,nj->ni", camera_rot_t, points_world - camera_pos).astype(np.float64)


def load_body_points_from_processed(processed_demo_dir: Path, side: str, part: str) -> np.ndarray:
    manifest = json.loads((processed_demo_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    indices = np.asarray(manifest["selected_source_indices"], dtype=np.int64)
    with h5py.File(str(manifest["source_hdf5"]), "r") as h5:
        points_world = h5[f"transforms/{side}{part}"][indices, :3, 3].astype(np.float64)
        camera_transforms = h5["transforms/camera"][indices].astype(np.float64)
    return world_points_to_camera(points_world, camera_transforms)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def base_quat(root: np.ndarray, target: np.ndarray, side: str, axis: str) -> np.ndarray:
    forward = normalize(target - root)
    if axis == "x":
        x_axis = forward
        y_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        if side == "right":
            y_hint = -y_hint
        z_axis = np.cross(x_axis, y_hint)
        if np.linalg.norm(z_axis) < 1e-6:
            z_axis = np.cross(x_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        z_axis = normalize(z_axis)
        y_axis = normalize(np.cross(z_axis, x_axis))
        mat = np.column_stack([x_axis, y_axis, z_axis])
    elif axis == "minus_z":
        z_axis = -forward
        x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float64) if side == "left" else np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_hint)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.cross(z_axis, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        y_axis = normalize(y_axis)
        x_axis = normalize(np.cross(y_axis, z_axis))
        mat = np.column_stack([x_axis, y_axis, z_axis])
    else:
        z_axis = forward
        x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float64) if side == "left" else np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_hint)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.cross(z_axis, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        y_axis = normalize(y_axis)
        x_axis = normalize(np.cross(y_axis, z_axis))
        mat = np.column_stack([x_axis, y_axis, z_axis])
    return R.from_matrix(mat).as_quat(scalar_first=True)


def style_kinova_robot(model) -> None:
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


def configure_camera_space_zed(model, fovy: float) -> str:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed")
    if cam_id < 0:
        raise ValueError("Compiled Phantom XML does not contain a zed camera.")
    model.cam_pos[cam_id] = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    # MuJoCo cameras look along local -Z. This rotation maps local -Z to
    # camera-space +Z and local +Y to image up (-camera Y).
    model.cam_quat[cam_id] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    model.cam_fovy[cam_id] = float(fovy)
    return "zed"


def render_rgb_mask_depth_camera(renderer, model, data, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return rgb, mask, depth


def load_scene_depth(path: Path, n_frames: int, height: int, width: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Depth must have shape (T,H,W), got {depth.shape}: {path}")
    if len(depth) < n_frames:
        raise ValueError(f"Depth length {len(depth)} is shorter than frames {n_frames}: {path}")
    depth = depth[:n_frames]
    if not np.isfinite(depth).all():
        raise ValueError(f"Depth contains non-finite values: {path}")
    if float(depth.std()) < 1e-5:
        raise ValueError(f"Depth appears constant; refusing depth-aware overlay: {path}")
    if depth.shape[1:3] != (height, width):
        depth = np.asarray(
            [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth],
            dtype=np.float32,
        )
    return depth


def overlay_robot_with_depth(
    raw: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    scene_depth: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene_depth.shape != robot_depth.shape:
        scene_depth = cv2.resize(scene_depth.astype(np.float32), robot_depth.shape[::-1], interpolation=cv2.INTER_LINEAR)
    valid_depth = np.isfinite(robot_depth) & np.isfinite(scene_depth)
    visible = robot_mask & valid_depth & (robot_depth <= scene_depth + float(margin))
    occluded = robot_mask & ~visible
    out = raw.copy()
    out[visible] = robot_rgb[visible]
    return out, visible.astype(bool), occluded.astype(bool)


def set_initial_kinova_qpos(model, data) -> None:
    data.qpos[:] = 0.0
    for idx, value in enumerate(KINOVA_INIT_QPOS, start=1):
        set_joint_qpos(mujoco, model, data, f"robot0_Actuator{idx}", float(value))
        set_joint_qpos(mujoco, model, data, f"robot1_Actuator{idx}", float(value))
    mujoco.mj_forward(model, data)


def set_robotiq_width(model, data, prefix: str, width: float) -> None:
    opening = float(np.clip(width / 0.085, 0.0, 1.0))
    driver = np.interp(opening, [0.0, 1.0], [0.9, 0.0])
    for joint in (
        f"{prefix}_left_driver_joint",
        f"{prefix}_left_spring_link_joint",
        f"{prefix}_left_follower",
        f"{prefix}_right_driver_joint",
        f"{prefix}_right_spring_link_joint",
        f"{prefix}_right_follower_joint",
    ):
        set_joint_qpos(mujoco, model, data, joint, float(driver))


def ik_position_with_link_targets(
    model,
    data,
    *,
    ee_target: np.ndarray,
    ee_body_id: int,
    link_targets: list[tuple[int, np.ndarray, float]],
    initial_qpos: np.ndarray | None,
    tol: float,
    link_tol: float,
    max_iters: int,
    active_joint_prefixes: tuple[str, ...],
) -> tuple[bool, np.ndarray, float, float]:
    data.qpos[:] = 0.0 if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64)
    clamp_qpos(model, data.qpos)
    mujoco.mj_forward(model, data)
    ee_target = np.asarray(ee_target, dtype=np.float64).reshape(3)
    dof_mask = active_dof_mask(mujoco, model, active_joint_prefixes)
    damping = 1e-4
    for _ in range(max_iters):
        rows = []
        residuals = []
        ee_pos = body_point(data, ee_body_id, np.zeros(3))
        ee_err = ee_target - ee_pos
        ee_err_norm = float(np.linalg.norm(ee_err))
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, jacr, ee_pos, ee_body_id)
        rows.append(jacp)
        residuals.append(ee_err)
        aux_errors = []
        for link_body_id, target, weight in link_targets:
            if weight <= 0.0:
                continue
            link_pos = body_point(data, link_body_id, np.zeros(3))
            err = np.asarray(target, dtype=np.float64).reshape(3) - link_pos
            aux_errors.append(float(np.linalg.norm(err)))
            jacp_link = np.zeros((3, model.nv), dtype=np.float64)
            jacr_link = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(model, data, jacp_link, jacr_link, link_pos, link_body_id)
            rows.append(float(weight) * jacp_link)
            residuals.append(float(weight) * err)
        aux_err = float(np.mean(aux_errors)) if aux_errors else 0.0
        if ee_err_norm < tol and (not aux_errors or aux_err < link_tol):
            return True, data.qpos.copy(), ee_err_norm, aux_err
        jac = np.vstack(rows)
        residual = np.concatenate(residuals)
        lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(lhs, residual)
        dq = dq * dof_mask
        mujoco.mj_integratePos(model, data.qpos, dq, 0.35)
        clamp_qpos(model, data.qpos)
        mujoco.mj_forward(model, data)
    ee_err_norm = float(np.linalg.norm(ee_target - body_point(data, ee_body_id, np.zeros(3))))
    aux_errors = [
        float(np.linalg.norm(np.asarray(target, dtype=np.float64).reshape(3) - body_point(data, bid, np.zeros(3))))
        for bid, target, weight in link_targets
        if weight > 0.0
    ]
    return ee_err_norm < tol, data.qpos.copy(), ee_err_norm, float(np.mean(aux_errors)) if aux_errors else 0.0


def human_link_targets(
    body_points_camera: dict[str, dict[str, np.ndarray]],
    side: str,
    frame_i: int,
    *,
    root: np.ndarray,
    half_arm_body_id: int,
    forearm_body_id: int,
    wrist_body_id: int,
    weights: tuple[float, float, float],
) -> list[tuple[int, np.ndarray, float]]:
    forearm = body_points_camera[side]["Forearm"][frame_i].astype(np.float64)
    hand = body_points_camera[side]["Hand"][frame_i].astype(np.float64)
    if float(forearm[2]) < 0.08:
        forearm = 0.45 * forearm + 0.55 * hand
        forearm[2] = max(float(forearm[2]), 0.08)
    upper = 0.55 * np.asarray(root, dtype=np.float64) + 0.45 * forearm
    return [
        (half_arm_body_id, upper, weights[0]),
        (forearm_body_id, forearm, weights[1]),
        (wrist_body_id, hand, weights[2]),
    ]


def make_montage(frames: list[np.ndarray], output_path: Path) -> None:
    ids = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(np.concatenate([frames[i] for i in ids], axis=1), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render original Phantom Kinova3 on EgoDex with camera-space IK.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--setup", type=str, default="shoulders")
    parser.add_argument("--compiled-xml", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--ik-tol", type=float, default=0.035)
    parser.add_argument("--link-tol", type=float, default=0.08)
    parser.add_argument("--base-axis", choices=("x", "z", "minus_z"), default="minus_z")
    parser.add_argument("--root-border-pad-px", type=float, default=24.0)
    parser.add_argument("--root-depth-offset-m", type=float, default=0.10)
    parser.add_argument("--half-arm-weight", type=float, default=0.18)
    parser.add_argument("--forearm-weight", type=float, default=0.45)
    parser.add_argument("--wrist-weight", type=float, default=0.15)
    parser.add_argument("--depth-aware-overlay", action="store_true")
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--depth-occlusion-margin", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    output_dir = (args.output_dir or processed_demo_dir / "kinova_camera_ik").resolve()
    xml_path = args.compiled_xml or output_dir / "phantom_kinova3_robotiq85_shoulders.xml"
    export_phantom_kinova_xml(xml_path)

    raw_frames = read_video_frames(processed_demo_dir / "video_rgb_imgs.mkv")
    if args.max_frames is not None:
        raw_frames = raw_frames[: args.max_frames]
    intrinsic = read_intrinsic(processed_demo_dir)
    traj = build_traj_like(processed_demo_dir, args.setup, intrinsic)
    left_targets = np.asarray(traj["left_ee_pos_camera"], dtype=np.float64)[: len(raw_frames)]
    right_targets = np.asarray(traj["right_ee_pos_camera"], dtype=np.float64)[: len(raw_frames)]
    left_width = np.asarray(traj["left_width"], dtype=np.float64)[: len(raw_frames)]
    right_width = np.asarray(traj["right_width"], dtype=np.float64)[: len(raw_frames)]
    height, width = raw_frames[0].shape[:2]
    scene_depth = None
    if args.depth_aware_overlay:
        depth_path = args.depth_path.resolve() if args.depth_path is not None else processed_demo_dir / "depth.npy"
        scene_depth = load_scene_depth(depth_path, len(raw_frames), height, width)

    body_points = {
        side: {part: load_body_points_from_processed(processed_demo_dir, side, part)[: len(raw_frames)] for part in ("Forearm", "Hand", "Arm")}
        for side in ("left", "right")
    }
    left_root, left_root_pixel = fixed_root_from_forearm_border(
        left_targets,
        body_points["left"]["Forearm"],
        body_points["left"]["Arm"],
        intrinsic,
        width,
        height,
        "left",
        border_pad_px=args.root_border_pad_px,
        depth_offset_m=args.root_depth_offset_m,
    )
    right_root, right_root_pixel = fixed_root_from_forearm_border(
        right_targets,
        body_points["right"]["Forearm"],
        body_points["right"]["Arm"],
        intrinsic,
        width,
        height,
        "right",
        border_pad_px=args.root_border_pad_px,
        depth_offset_m=args.root_depth_offset_m,
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    style_kinova_robot(model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera_name = configure_camera_space_zed(model, fovy_from_intrinsic(intrinsic, height))
    model.body_pos[body_id(mujoco, model, "robot0_base")] = right_root
    model.body_quat[body_id(mujoco, model, "robot0_base")] = base_quat(
        right_root, np.median(right_targets, axis=0), "right", args.base_axis
    )
    model.body_pos[body_id(mujoco, model, "robot1_base")] = left_root
    model.body_quat[body_id(mujoco, model, "robot1_base")] = base_quat(
        left_root, np.median(left_targets, axis=0), "left", args.base_axis
    )

    right_ee_id = body_id(mujoco, model, "gripper0_eef")
    left_ee_id = body_id(mujoco, model, "gripper1_eef")
    ids = {
        "right": (
            body_id(mujoco, model, "robot0_HalfArm2_Link"),
            body_id(mujoco, model, "robot0_forearm_link"),
            body_id(mujoco, model, "robot0_Bracelet_Link"),
        ),
        "left": (
            body_id(mujoco, model, "robot1_HalfArm2_Link"),
            body_id(mujoco, model, "robot1_forearm_link"),
            body_id(mujoco, model, "robot1_Bracelet_Link"),
        ),
    }
    renderer = mujoco.Renderer(model, height=height, width=width)
    set_initial_kinova_qpos(model, data)
    previous_qpos = data.qpos.copy()
    robot_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    robot_masks: list[np.ndarray] = []
    robot_depths: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []
    occlusion_masks: list[np.ndarray] = []
    ik_errors = []
    link_weights = (args.half_arm_weight, args.forearm_weight, args.wrist_weight)
    for frame_i in tqdm(range(len(raw_frames)), desc="Render Kinova3"):
        set_robotiq_width(model, data, "gripper0", right_width[frame_i])
        set_robotiq_width(model, data, "gripper1", left_width[frame_i])
        ok_right, qpos, err_right, aux_right = ik_position_with_link_targets(
            model,
            data,
            ee_target=right_targets[frame_i],
            ee_body_id=right_ee_id,
            link_targets=human_link_targets(
                body_points,
                "right",
                frame_i,
                root=right_root,
                half_arm_body_id=ids["right"][0],
                forearm_body_id=ids["right"][1],
                wrist_body_id=ids["right"][2],
                weights=link_weights,
            ),
            initial_qpos=previous_qpos,
            tol=args.ik_tol,
            link_tol=args.link_tol,
            max_iters=180,
            active_joint_prefixes=("robot0_Actuator",),
        )
        ok_left, qpos, err_left, aux_left = ik_position_with_link_targets(
            model,
            data,
            ee_target=left_targets[frame_i],
            ee_body_id=left_ee_id,
            link_targets=human_link_targets(
                body_points,
                "left",
                frame_i,
                root=left_root,
                half_arm_body_id=ids["left"][0],
                forearm_body_id=ids["left"][1],
                wrist_body_id=ids["left"][2],
                weights=link_weights,
            ),
            initial_qpos=qpos,
            tol=args.ik_tol,
            link_tol=args.link_tol,
            max_iters=180,
            active_joint_prefixes=("robot1_Actuator",),
        )
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        previous_qpos = data.qpos.copy()
        rgb, mask, robot_depth = render_rgb_mask_depth_camera(renderer, model, data, camera_name)
        robot_frames.append(rgb)
        robot_masks.append(mask)
        robot_depths.append(robot_depth)
        if scene_depth is not None:
            overlay, visible_mask, occlusion_mask = overlay_robot_with_depth(
                raw_frames[frame_i],
                rgb,
                mask,
                robot_depth,
                scene_depth[frame_i],
                args.depth_occlusion_margin,
            )
            overlay_frames.append(overlay)
            visible_masks.append(visible_mask)
            occlusion_masks.append(occlusion_mask)
        else:
            overlay_frames.append(overlay_robot(raw_frames[frame_i], rgb, mask))
        ik_errors.append([float(err_left), float(err_right), float(aux_left), float(aux_right), bool(ok_left), bool(ok_right)])
    renderer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    robot_video = output_dir / "video_robot_Kinova3_camera_ik.mp4"
    overlay_video = output_dir / "video_overlay_Kinova3_camera_ik_rawhand.mp4"
    robot_mask_npz = output_dir / "robot_masks_Kinova3_camera_ik.npz"
    montage = output_dir / "kinova_camera_ik_montage.jpg"
    save_video(robot_frames, robot_video, args.fps)
    save_video(overlay_frames, overlay_video, args.fps)
    np.savez_compressed(
        robot_mask_npz,
        robot_mask=np.asarray(robot_masks, dtype=bool),
        robot_depth=np.asarray(robot_depths, dtype=np.float32),
        visible_mask=np.asarray(visible_masks, dtype=bool) if visible_masks else np.zeros((0,), dtype=bool),
        occlusion_mask=np.asarray(occlusion_masks, dtype=bool) if occlusion_masks else np.zeros((0,), dtype=bool),
    )
    make_montage(overlay_frames, montage)
    errors = np.asarray(ik_errors, dtype=object)
    manifest = {
        "stage": "egodex_original_action_kinova_camera_ik",
        "processed_demo_dir": str(processed_demo_dir),
        "compiled_xml": str(xml_path),
        "robot_video": str(robot_video),
        "rawhand_overlay_video": str(overlay_video),
        "robot_mask_npz": str(robot_mask_npz),
        "montage": str(montage),
        "left_root_camera": left_root.tolist(),
        "right_root_camera": right_root.tolist(),
        "left_root_pixel": list(left_root_pixel),
        "right_root_pixel": list(right_root_pixel),
        "base_axis": args.base_axis,
        "depth_aware_overlay": bool(args.depth_aware_overlay),
        "depth_path": str(args.depth_path.resolve() if args.depth_path is not None else processed_demo_dir / "depth.npy") if args.depth_aware_overlay else None,
        "depth_occlusion_margin": float(args.depth_occlusion_margin),
        "mean_visible_robot_area": float(np.mean([m.mean() for m in visible_masks])) if visible_masks else None,
        "mean_occluded_robot_area": float(np.mean([m.mean() for m in occlusion_masks])) if occlusion_masks else None,
        "mean_left_ik_error": float(np.mean(errors[:, 0].astype(float))),
        "mean_right_ik_error": float(np.mean(errors[:, 1].astype(float))),
        "mean_left_link_error": float(np.mean(errors[:, 2].astype(float))),
        "mean_right_link_error": float(np.mean(errors[:, 3].astype(float))),
        "failed_left_frames": int(np.sum(~errors[:, 4].astype(bool))),
        "failed_right_frames": int(np.sum(~errors[:, 5].astype(bool))),
    }
    write_json(output_dir / "kinova_camera_ik_manifest.json", manifest)
    print(output_dir / "kinova_camera_ik_manifest.json")


if __name__ == "__main__":
    main()
