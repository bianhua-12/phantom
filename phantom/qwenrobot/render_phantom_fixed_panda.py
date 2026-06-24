from __future__ import annotations

import argparse
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

from phantom.qwenrobot.common import DEFAULT_FPS, DEFAULT_OUTPUT_DIR, read_json, safe_id, save_video, video_info, write_json
from phantom.qwenrobot.mujoco_utils import active_dof_mask, body_id, body_point, clamp_qpos, ik_position, set_joint_qpos


PHANTOM_ROBOSUITE = Path(__file__).resolve().parents[2] / "submodules" / "phantom-robosuite"
PHANTOM_ROBOMIMIC = Path(__file__).resolve().parents[2] / "submodules" / "phantom-robomimic"

PANDA_INIT_QPOS = np.asarray(
    [0.0, math.pi / 16.0, 0.0, -math.pi / 2.0 - math.pi / 3.0, 0.0, math.pi - 0.2, math.pi / 4.0],
    dtype=np.float64,
)


def ik_position_with_aux(
    mujoco_module,
    model,
    data,
    target: np.ndarray,
    ee_body_id: int,
    *,
    aux_body_id: int | None,
    aux_target: np.ndarray | None,
    aux_weight: float,
    initial_qpos: np.ndarray | None = None,
    tol: float = 0.03,
    max_iters: int = 180,
    active_joint_prefixes: tuple[str, ...] | None = None,
) -> tuple[bool, np.ndarray, float]:
    if aux_body_id is None or aux_target is None or aux_weight <= 0.0:
        return ik_position(
            mujoco_module,
            model,
            data,
            target,
            ee_body_id,
            initial_qpos=initial_qpos,
            tol=tol,
            max_iters=max_iters,
            active_joint_prefixes=active_joint_prefixes,
            orientation_weight=0.0,
        )

    data.qpos[:] = 0.0 if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64)
    clamp_qpos(model, data.qpos)
    mujoco_module.mj_forward(model, data)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    aux_target = np.asarray(aux_target, dtype=np.float64).reshape(3)
    dof_mask = active_dof_mask(mujoco_module, model, active_joint_prefixes)
    damping = 1e-4
    for _ in range(max_iters):
        ee_pos = body_point(data, ee_body_id, np.zeros(3))
        aux_pos = body_point(data, aux_body_id, np.zeros(3))
        ee_err = target - ee_pos
        aux_err = aux_target - aux_pos
        pos_err = float(np.linalg.norm(ee_err))
        if pos_err < tol:
            return True, data.qpos.copy(), pos_err
        jacp_ee = np.zeros((3, model.nv), dtype=np.float64)
        jacr_ee = np.zeros((3, model.nv), dtype=np.float64)
        jacp_aux = np.zeros((3, model.nv), dtype=np.float64)
        jacr_aux = np.zeros((3, model.nv), dtype=np.float64)
        mujoco_module.mj_jac(model, data, jacp_ee, jacr_ee, ee_pos, ee_body_id)
        mujoco_module.mj_jac(model, data, jacp_aux, jacr_aux, aux_pos, aux_body_id)
        jac = np.vstack([jacp_ee, aux_weight * jacp_aux])
        residual = np.concatenate([ee_err, aux_weight * aux_err])
        lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(lhs, residual)
        if dof_mask is not None:
            dq = dq * dof_mask
        mujoco_module.mj_integratePos(model, data.qpos, dq, 0.35)
        clamp_qpos(model, data.qpos)
        mujoco_module.mj_forward(model, data)
    pos_err = float(np.linalg.norm(target - body_point(data, ee_body_id, np.zeros(3))))
    return pos_err < tol, data.qpos.copy(), pos_err


def _add_phantom_submodules_to_path() -> None:
    for path in (PHANTOM_ROBOSUITE, PHANTOM_ROBOMIMIC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def export_phantom_panda_xml(xml_path: Path) -> None:
    if xml_path.exists():
        return
    _add_phantom_submodules_to_path()
    from robosuite.controllers import load_controller_config
    from robomimic.envs.env_robosuite import EnvRobosuite
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=["frontview_image"])))
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["control_delta"] = False
    controller_config["uncouple_pos_ori"] = False
    options = dict(
        env_name="PhantomBimanual",
        bimanual_setup="shoulders",
        robots=["Panda", "Panda"],
        gripper_types=["Robotiq85Gripper", "Robotiq85Gripper"],
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


def read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def fovy_from_intrinsic(intrinsic: np.ndarray, height: int) -> float:
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * float(intrinsic[1, 1]))))


def project_camera_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = np.maximum(points[:, 2], 1e-6)
    return np.column_stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy])


def backproject_pixel(pixel_u: float, pixel_v: float, depth: float, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    return np.asarray([(pixel_u - cx) / fx * depth, (pixel_v - cy) / fy * depth, depth], dtype=np.float64)


def world_points_to_camera(points_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    camera_pos = camera_transforms[:, :3, 3]
    camera_rot = camera_transforms[:, :3, :3]
    camera_rot_t = np.swapaxes(camera_rot, 1, 2)
    return np.einsum("nij,nj->ni", camera_rot_t, points_world - camera_pos).astype(np.float64)


def load_body_points_camera(traj: np.lib.npyio.NpzFile, side: str, part: str) -> np.ndarray | None:
    cached_key = f"{side}_{part.lower()}_pos_camera"
    if cached_key in traj.files:
        return traj[cached_key].astype(np.float64)
    if "source_hdf5" not in traj.files or "frame_indices" not in traj.files:
        return None
    source_hdf5 = Path(str(traj["source_hdf5"].item() if traj["source_hdf5"].shape == () else traj["source_hdf5"]))
    key = f"transforms/{side}{part}"
    with h5py.File(source_hdf5, "r") as h5:
        if key not in h5 or "transforms/camera" not in h5:
            return None
        indices = traj["frame_indices"].astype(np.int64)
        points_world = h5[key][indices, :3, 3].astype(np.float64)
        camera_transforms = h5["transforms/camera"][indices].astype(np.float64)
    return world_points_to_camera(points_world, camera_transforms)


def ray_to_padded_image_border(anchor: np.ndarray, direction: np.ndarray, width: int, height: int, pad: float) -> np.ndarray:
    direction = normalize(direction[:2])
    x0, y0 = float(anchor[0]), float(anchor[1])
    candidates: list[tuple[float, np.ndarray]] = []
    bounds = (-pad, float(width) + pad, -pad, float(height) + pad)
    min_x, max_x, min_y, max_y = bounds
    if abs(float(direction[0])) > 1e-6:
        for x in (min_x, max_x):
            t = (x - x0) / float(direction[0])
            y = y0 + t * float(direction[1])
            if t > 0.0 and min_y - 1e-4 <= y <= max_y + 1e-4:
                candidates.append((t, np.asarray([x, y], dtype=np.float64)))
    if abs(float(direction[1])) > 1e-6:
        for y in (min_y, max_y):
            t = (y - y0) / float(direction[1])
            x = x0 + t * float(direction[0])
            if t > 0.0 and min_x - 1e-4 <= x <= max_x + 1e-4:
                candidates.append((t, np.asarray([x, y], dtype=np.float64)))
    if not candidates:
        return np.asarray(anchor, dtype=np.float64)
    return min(candidates, key=lambda item: item[0])[1]


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def base_quat_toward_target(root: np.ndarray, target: np.ndarray, side: str) -> np.ndarray:
    z_axis = normalize(target - root)
    x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float64) if side == "left" else np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_hint)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(z_axis, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    y_axis = normalize(y_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat(scalar_first=True)


def fixed_root_from_hand_pixels(
    hand_points_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    side: str,
    *,
    horizontal_offset_px: float,
    bottom_offset_px: float,
    depth_offset_m: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    pixels = project_camera_points(hand_points_camera, intrinsic)
    median_pixel = np.median(pixels, axis=0)
    root_u = float(np.clip(median_pixel[0] + (-horizontal_offset_px if side == "left" else horizontal_offset_px), 80.0, image_width - 80.0))
    root_v = float(image_height + bottom_offset_px)
    root_depth = float(np.median(hand_points_camera[:, 2]) + depth_offset_m)
    return backproject_pixel(root_u, root_v, root_depth, intrinsic), (root_u, root_v)


def fixed_root_from_forearm_border(
    hand_points_camera: np.ndarray,
    forearm_points_camera: np.ndarray | None,
    arm_points_camera: np.ndarray | None,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    side: str,
    *,
    border_pad_px: float,
    depth_offset_m: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    hand_pixels = project_camera_points(hand_points_camera, intrinsic)
    source_points = forearm_points_camera
    if source_points is None or not np.isfinite(source_points).all():
        source_points = arm_points_camera
    if source_points is None:
        source_pixels = np.full_like(hand_pixels, np.nan)
        source_depth = hand_points_camera[:, 2]
    else:
        source_pixels = project_camera_points(source_points, intrinsic)
        source_depth = source_points[:, 2]

    fallback = np.asarray([-1.0, 0.65], dtype=np.float64) if side == "left" else np.asarray([1.0, 0.65], dtype=np.float64)
    root_pixels = []
    root_depths = []
    for hand_px, source_px, hand_z, source_z in zip(hand_pixels, source_pixels, hand_points_camera[:, 2], source_depth):
        if not np.isfinite(hand_px).all() or hand_z <= 0.05:
            continue
        direction = source_px - hand_px if np.isfinite(source_px).all() and source_z > 0.05 else fallback
        if np.linalg.norm(direction) < 1e-4:
            direction = fallback
        root_pixels.append(ray_to_padded_image_border(hand_px, direction, image_width, image_height, border_pad_px))
        if np.isfinite(source_z) and source_z > 0.05:
            root_depths.append(float(source_z + depth_offset_m))
        else:
            root_depths.append(float(hand_z + depth_offset_m))

    if not root_pixels:
        return fixed_root_from_hand_pixels(
            hand_points_camera,
            intrinsic,
            image_width,
            image_height,
            side,
            horizontal_offset_px=border_pad_px,
            bottom_offset_px=border_pad_px,
            depth_offset_m=depth_offset_m,
        )
    root_pixel = np.median(np.asarray(root_pixels, dtype=np.float64), axis=0)
    root_depth = float(np.clip(np.median(root_depths), 0.08, 1.5))
    return backproject_pixel(float(root_pixel[0]), float(root_pixel[1]), root_depth, intrinsic), (
        float(root_pixel[0]),
        float(root_pixel[1]),
    )


def style_phantom_robot(mujoco_module, model) -> None:
    hidden = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    robot_rgba = np.asarray([0.82, 0.84, 0.86, 1.0], dtype=np.float32)
    gripper_rgba = np.asarray([0.08, 0.08, 0.08, 1.0], dtype=np.float32)
    for geom_id in range(model.ngeom):
        body_name = mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        is_robot = body_name.startswith("robot") or body_name.startswith("gripper")
        if not is_robot:
            model.geom_rgba[geom_id] = hidden
        elif model.geom_type[geom_id] != int(mujoco_module.mjtGeom.mjGEOM_MESH) and "pad" not in body_name:
            model.geom_rgba[geom_id] = hidden
        elif body_name.startswith("gripper") or "finger" in body_name or "pad" in body_name:
            model.geom_rgba[geom_id] = gripper_rgba
        else:
            model.geom_rgba[geom_id] = robot_rgba
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]


def set_initial_panda_qpos(mujoco_module, model, data) -> None:
    data.qpos[:] = 0.0
    for idx, value in enumerate(PANDA_INIT_QPOS, start=1):
        set_joint_qpos(mujoco_module, model, data, f"robot0_joint{idx}", float(value))
        set_joint_qpos(mujoco_module, model, data, f"robot1_joint{idx}", float(value))
    mujoco_module.mj_forward(model, data)


def render_rgb_and_mask(mujoco_module, renderer, model, data, camera) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    scene_camera = renderer._scene.camera[0]
    scene_camera.pos[:] = [0.0, 0.0, 0.0]
    scene_camera.forward[:] = [0.0, 0.0, 1.0]
    scene_camera.up[:] = [0.0, -1.0, 0.0]
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    scene_camera = renderer._scene.camera[0]
    scene_camera.pos[:] = [0.0, 0.0, 0.0]
    scene_camera.forward[:] = [0.0, 0.0, 1.0]
    scene_camera.up[:] = [0.0, -1.0, 0.0]
    seg = renderer.render()
    geom_ids = seg[:, :, 0].astype(np.int32)
    mask = np.zeros(geom_ids.shape, dtype=bool)
    for geom_id in np.unique(geom_ids):
        if geom_id < 0 or geom_id >= model.ngeom:
            continue
        body_name = mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("robot") or body_name.startswith("gripper"):
            mask |= geom_ids == geom_id
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return rgb, mask


def overlay_robot(raw: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[robot_mask] = robot_rgb[robot_mask]
    return out


def valid_aux_target(points: np.ndarray | None, frame_i: int, root: np.ndarray, target: np.ndarray) -> np.ndarray:
    if points is not None and frame_i < len(points):
        point = points[frame_i].astype(np.float64)
        if np.isfinite(point).all() and point[2] > 0.05:
            return point
    return 0.55 * np.asarray(root, dtype=np.float64) + 0.45 * np.asarray(target, dtype=np.float64)


def make_montage(frames: list[np.ndarray], output_path: Path) -> None:
    if not frames:
        return
    ids = [0, len(frames) // 2, len(frames) - 1]
    montage = np.concatenate([frames[i] for i in ids], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render EgoDex demos with Phantom's original fixed-shoulder Panda robot in camera coordinates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiled-xml", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--ik-tol", type=float, default=0.015)
    parser.add_argument("--rot-tol", type=float, default=0.75)
    parser.add_argument("--orientation-weight", type=float, default=0.0)
    parser.add_argument("--root-strategy", choices=["forearm_border", "fixed_offset"], default="forearm_border")
    parser.add_argument("--horizontal-root-offset-px", type=float, default=120.0)
    parser.add_argument("--bottom-root-offset-px", type=float, default=80.0)
    parser.add_argument("--root-border-pad-px", type=float, default=96.0)
    parser.add_argument("--root-depth-offset-m", type=float, default=0.25)
    parser.add_argument("--forearm-aux-weight", type=float, default=0.0)
    parser.add_argument("--left-aux-body", type=str, default="robot0_link5")
    parser.add_argument("--right-aux-body", type=str, default="robot1_link5")
    args = parser.parse_args()

    xml_path = args.compiled_xml or (args.output_dir / "05_phantom_assets" / "phantom_panda_robotiq_shoulders.xml")
    export_phantom_panda_xml(xml_path)

    manifest = read_json(args.output_dir / "00_manifest.json")
    rows = []
    for row in manifest["episodes"][: args.max_episodes]:
        sid = safe_id(row["id"])
        traj = np.load(row["trajectory_npz"], allow_pickle=True)
        if "left_ee_pos_camera" not in traj.files or "right_ee_pos_camera" not in traj.files:
            raise KeyError("Trajectory npz must contain left_ee_pos_camera and right_ee_pos_camera.")
        raw_frames = read_video_frames(Path(row["sampled_video"]))
        info = video_info(Path(row["sampled_video"]))
        width, height = int(info["width"]), int(info["height"])
        intrinsic = traj["camera_intrinsic"].astype(np.float64)

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        style_phantom_robot(mujoco, model)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        model.vis.global_.fovy = fovy_from_intrinsic(intrinsic, height)

        if args.root_strategy == "forearm_border":
            left_root, left_root_pixel = fixed_root_from_forearm_border(
                traj["left_ee_pos_camera"].astype(np.float64),
                load_body_points_camera(traj, "left", "Forearm"),
                load_body_points_camera(traj, "left", "Arm"),
                intrinsic,
                width,
                height,
                "left",
                border_pad_px=args.root_border_pad_px,
                depth_offset_m=args.root_depth_offset_m,
            )
            right_root, right_root_pixel = fixed_root_from_forearm_border(
                traj["right_ee_pos_camera"].astype(np.float64),
                load_body_points_camera(traj, "right", "Forearm"),
                load_body_points_camera(traj, "right", "Arm"),
                intrinsic,
                width,
                height,
                "right",
                border_pad_px=args.root_border_pad_px,
                depth_offset_m=args.root_depth_offset_m,
            )
        else:
            left_root, left_root_pixel = fixed_root_from_hand_pixels(
                traj["left_ee_pos_camera"].astype(np.float64),
                intrinsic,
                width,
                height,
                "left",
                horizontal_offset_px=args.horizontal_root_offset_px,
                bottom_offset_px=args.bottom_root_offset_px,
                depth_offset_m=args.root_depth_offset_m,
            )
            right_root, right_root_pixel = fixed_root_from_hand_pixels(
                traj["right_ee_pos_camera"].astype(np.float64),
                intrinsic,
                width,
                height,
                "right",
                horizontal_offset_px=args.horizontal_root_offset_px,
                bottom_offset_px=args.bottom_root_offset_px,
                depth_offset_m=args.root_depth_offset_m,
            )
        left_target_median = np.median(traj["left_ee_pos_camera"].astype(np.float64), axis=0)
        right_target_median = np.median(traj["right_ee_pos_camera"].astype(np.float64), axis=0)
        left_base_id = body_id(mujoco, model, "robot0_base")
        right_base_id = body_id(mujoco, model, "robot1_base")
        model.body_pos[left_base_id] = left_root
        model.body_quat[left_base_id] = base_quat_toward_target(left_root, left_target_median, "left")
        model.body_pos[right_base_id] = right_root
        model.body_quat[right_base_id] = base_quat_toward_target(right_root, right_target_median, "right")

        left_ee_id = body_id(mujoco, model, "gripper0_eef")
        right_ee_id = body_id(mujoco, model, "gripper1_eef")
        left_aux_body_id = body_id(mujoco, model, args.left_aux_body) if args.forearm_aux_weight > 0 else None
        right_aux_body_id = body_id(mujoco, model, args.right_aux_body) if args.forearm_aux_weight > 0 else None
        left_forearm_camera = load_body_points_camera(traj, "left", "Forearm")
        right_forearm_camera = load_body_points_camera(traj, "right", "Forearm")
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        set_initial_panda_qpos(mujoco, model, data)
        previous_qpos = data.qpos.copy()
        robot_frames: list[np.ndarray] = []
        overlay_frames: list[np.ndarray] = []
        qpos_rows = []
        ik_errors = []
        num_frames = min(len(raw_frames), len(traj["left_ee_pos_camera"]), len(traj["right_ee_pos_camera"]))
        for frame_i in tqdm(range(num_frames), desc=f"Render {sid}"):
            left_target = traj["left_ee_pos_camera"][frame_i].astype(np.float64)
            right_target = traj["right_ee_pos_camera"][frame_i].astype(np.float64)
            ok_left, qpos, err_left = ik_position_with_aux(
                mujoco,
                model,
                data,
                left_target,
                left_ee_id,
                aux_body_id=left_aux_body_id,
                aux_target=valid_aux_target(left_forearm_camera, frame_i, left_root, left_target),
                aux_weight=args.forearm_aux_weight,
                initial_qpos=previous_qpos,
                tol=args.ik_tol,
                max_iters=180,
                active_joint_prefixes=("robot0_joint",),
            )
            ok_right, qpos, err_right = ik_position_with_aux(
                mujoco,
                model,
                data,
                right_target,
                right_ee_id,
                aux_body_id=right_aux_body_id,
                aux_target=valid_aux_target(right_forearm_camera, frame_i, right_root, right_target),
                aux_weight=args.forearm_aux_weight,
                initial_qpos=qpos,
                tol=args.ik_tol,
                max_iters=180,
                active_joint_prefixes=("robot1_joint",),
            )
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            previous_qpos = data.qpos.copy()
            qpos_rows.append(previous_qpos)
            ik_errors.append([float(err_left), float(err_right), bool(ok_left), bool(ok_right)])
            rgb, mask = render_rgb_and_mask(mujoco, renderer, model, data, camera)
            robot_frames.append(rgb)
            overlay_frames.append(overlay_robot(raw_frames[frame_i], rgb, mask))
        renderer.close()

        robot_video = args.output_dir / "05_phantom_fixed_panda_robot" / f"{sid}.mp4"
        overlay_video = args.output_dir / "05_phantom_fixed_panda_overlay" / f"{sid}.mp4"
        qpos_npz = args.output_dir / "05_phantom_fixed_panda_qpos" / f"{sid}.npz"
        save_video(robot_frames, robot_video, DEFAULT_FPS)
        save_video(overlay_frames, overlay_video, DEFAULT_FPS)
        make_montage(overlay_frames, args.output_dir / "05_phantom_fixed_panda_montage" / f"{sid}.jpg")
        qpos_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            qpos_npz,
            qpos=np.asarray(qpos_rows, dtype=np.float32),
            ik_errors=np.asarray(ik_errors, dtype=np.float32),
            left_root=left_root.astype(np.float32),
            right_root=right_root.astype(np.float32),
            left_root_pixel=np.asarray(left_root_pixel, dtype=np.float32),
            right_root_pixel=np.asarray(right_root_pixel, dtype=np.float32),
            xml_path=str(xml_path),
        )
        ik_arr = np.asarray(ik_errors, dtype=object)
        rows.append(
            {
                "id": row["id"],
                "robot_video": robot_video,
                "raw_robot_overlay_video": overlay_video,
                "qpos_npz": qpos_npz,
                "montage": args.output_dir / "05_phantom_fixed_panda_montage" / f"{sid}.jpg",
                "compiled_xml": xml_path,
                "left_root_camera": left_root,
                "right_root_camera": right_root,
                "left_root_pixel": left_root_pixel,
                "right_root_pixel": right_root_pixel,
                "root_strategy": args.root_strategy,
                "orientation_weight": args.orientation_weight,
                "rot_tol": args.rot_tol,
                "forearm_aux_weight": args.forearm_aux_weight,
                "left_aux_body": args.left_aux_body,
                "right_aux_body": args.right_aux_body,
                "mean_left_ik_error": float(np.mean([v[0] for v in ik_errors])),
                "mean_right_ik_error": float(np.mean([v[1] for v in ik_errors])),
                "failed_left_frames": int(np.sum([not v[2] for v in ik_errors])),
                "failed_right_frames": int(np.sum([not v[3] for v in ik_errors])),
            }
        )
    write_json(args.output_dir / "05_phantom_fixed_panda_manifest.json", {"stage": "phantom_original_panda_fixed_camera_roots", "episodes": rows})
    print(args.output_dir / "05_phantom_fixed_panda_manifest.json")


if __name__ == "__main__":
    main()
