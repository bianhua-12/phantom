from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from phantom.qwenrobot.common import save_video, write_json
from phantom.qwenrobot.mujoco_utils import (
    body_id,
    body_point,
    configure_camera_space_zed,
    export_phantom_kinova_xml,
    load_model,
    render_rgb_mask_depth,
    require_mujoco,
    set_initial_kinova_qpos,
    set_robotiq85_width,
    solve_arm_ik_selected,
    style_kinova_robot,
)
from phantom.qwenrobot.prepare_egodex_for_phantom import REPO_ROOT, load_extrinsics_matrix
from phantom.qwenrobot.render_phantom_fixed_panda import (
    fixed_root_from_forearm_border,
    fovy_from_intrinsic,
    overlay_robot,
    read_video_frames,
)


KEY_FRAMES = (27, 28, 29, 30, 31, 52, 75, 98)


def read_intrinsic(processed_demo_dir: Path) -> np.ndarray:
    candidates = [
        processed_demo_dir / "egodex_camera_intrinsics.json",
        processed_demo_dir.parents[2] / "egodex_camera_intrinsics.json",
        processed_demo_dir.parents[2] / "raw" / processed_demo_dir.parents[1].name / processed_demo_dir.name / "egodex_camera_intrinsics.json",
    ]
    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_dir = Path(str(manifest.get("raw_demo_dir", "")))
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
        path = Path(str(manifest.get("camera_extrinsics", "")))
        if path.exists():
            return path
    return REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json"


def actions_to_camera(points_action: np.ndarray, extrinsics_path: Path) -> np.ndarray:
    t_cam_to_action = load_extrinsics_matrix(extrinsics_path)
    t_action_to_cam = np.linalg.inv(t_cam_to_action)
    pts_h = np.concatenate([points_action, np.ones((len(points_action), 1), dtype=points_action.dtype)], axis=1)
    return (pts_h @ t_action_to_cam.T)[:, :3].astype(np.float64)


def rotations_to_camera(rot_action: np.ndarray, extrinsics_path: Path) -> np.ndarray:
    t_cam_to_action = load_extrinsics_matrix(extrinsics_path)
    r_action_to_cam = np.linalg.inv(t_cam_to_action[:3, :3])
    return np.einsum("ij,njk->nik", r_action_to_cam, rot_action).astype(np.float64)


def load_actions(processed_demo_dir: Path, setup: str) -> dict[str, np.ndarray]:
    smooth = processed_demo_dir / "smoothing_processor"
    extrinsics = camera_extrinsics_path(processed_demo_dir)
    left = np.load(smooth / f"smoothed_actions_left_{setup}.npz")
    right = np.load(smooth / f"smoothed_actions_right_{setup}.npz")
    return {
        "left_pos": actions_to_camera(left["ee_pts"].astype(np.float64), extrinsics),
        "right_pos": actions_to_camera(right["ee_pts"].astype(np.float64), extrinsics),
        "left_rot": rotations_to_camera(left["ee_oris"].astype(np.float64), extrinsics),
        "right_rot": rotations_to_camera(right["ee_oris"].astype(np.float64), extrinsics),
        "left_width": left["ee_widths"].astype(np.float64),
        "right_width": right["ee_widths"].astype(np.float64),
    }


def world_points_to_camera(points_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    camera_pos = camera_transforms[:, :3, 3]
    camera_rot = camera_transforms[:, :3, :3]
    camera_rot_t = np.swapaxes(camera_rot, 1, 2)
    return np.einsum("nij,nj->ni", camera_rot_t, points_world - camera_pos).astype(np.float64)


def load_hdf5_body_points(
    processed_demo_dir: Path,
    n_frames: int,
    source_hdf5: Path | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing adapter_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hdf5_path = source_hdf5.resolve() if source_hdf5 is not None else Path(str(manifest.get("source_hdf5", "")))
    indices = np.asarray(manifest.get("selected_source_indices", []), dtype=np.int64)
    try:
        if not hdf5_path.exists():
            raise FileNotFoundError(f"source_hdf5 not accessible: {hdf5_path}")
        out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        with h5py.File(str(hdf5_path), "r") as h5:
            camera = h5["transforms/camera"][indices].astype(np.float64)
            for side in ("left", "right"):
                for part in ("Forearm", "Hand", "Arm"):
                    key = f"transforms/{side}{part}"
                    points_world = h5[key][indices, :3, 3].astype(np.float64)
                    out[side][part] = world_points_to_camera(points_world, camera)[:n_frames]
        return out, str(hdf5_path)
    except Exception as exc:
        raise RuntimeError(
            "Cannot run Phantom-native explicit IK without EgoDex HDF5 body points "
            f"(Forearm/Hand/Arm/camera). source_hdf5={hdf5_path}"
        ) from exc


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
    if not np.isfinite(forearm).all() or float(forearm[2]) < 0.08:
        forearm = 0.45 * forearm + 0.55 * hand
        forearm[2] = max(float(forearm[2]), 0.08)
    upper = 0.55 * np.asarray(root, dtype=np.float64) + 0.45 * forearm
    return [
        (half_arm_body_id, upper, weights[0]),
        (forearm_body_id, forearm, weights[1]),
        (wrist_body_id, hand, weights[2]),
    ]


def load_background(processed_demo_dir: Path, explicit: Path | None) -> tuple[list[np.ndarray], str]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            processed_demo_dir / "inpaint_processor" / "video_human_inpaint_propainter_sam3_qwen_d5.mkv",
            processed_demo_dir / "video_rgb_imgs.mkv",
        ]
    )
    for path in candidates:
        if path.exists():
            return read_video_frames(path), str(path)
    raise FileNotFoundError(f"No background video found near {processed_demo_dir}")


def load_scene_depth(path: Path, n_frames: int, height: int, width: int) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Depth must have shape (T,H,W), got {depth.shape}: {path}")
    if len(depth) < n_frames:
        raise ValueError(f"Depth length {len(depth)} is shorter than {n_frames}: {path}")
    depth = depth[:n_frames]
    if depth.shape[1:3] != (height, width):
        depth = np.asarray([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth], dtype=np.float32)
    if not np.isfinite(depth).all() or float(depth.std()) < 1e-5:
        raise ValueError(f"Invalid scene depth: {path}")
    return depth


def overlay_with_depth(raw: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray, robot_depth: np.ndarray, scene_depth: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene_depth.shape != robot_depth.shape:
        scene_depth = cv2.resize(scene_depth.astype(np.float32), robot_depth.shape[::-1], interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(robot_depth) & np.isfinite(scene_depth) & (robot_depth > 0.0)
    visible = robot_mask & valid & (robot_depth <= scene_depth + float(margin))
    occluded = robot_mask & ~visible
    out = raw.copy()
    out[visible] = robot_rgb[visible]
    return out, visible.astype(bool), occluded.astype(bool)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[np.asarray(mask).astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def make_compare_montage(
    output: Path,
    frames: list[int],
    background: list[np.ndarray],
    reference_nodepth: list[np.ndarray] | None,
    reference_depth: list[np.ndarray] | None,
    robot_rgb: np.ndarray,
    nodepth: list[np.ndarray],
    scene_depth: np.ndarray | None,
    visible: np.ndarray,
    depth_overlay: list[np.ndarray] | None,
) -> None:
    rows = []
    for frame_i in frames:
        cells = [background[frame_i]]
        labels = ["clean"]
        if reference_nodepth is not None:
            cells.append(reference_nodepth[frame_i])
            labels.append("ref no-depth")
        if reference_depth is not None:
            cells.append(reference_depth[frame_i])
            labels.append("ref depth")
        cells.extend([robot_rgb[frame_i], nodepth[frame_i]])
        labels.extend(["canonical robot", "new no-depth"])
        if scene_depth is not None and depth_overlay is not None:
            cells.extend([colorize_depth(scene_depth[frame_i]), mask_rgb(visible[frame_i], (0, 255, 0)), depth_overlay[frame_i]])
            labels.extend(["scene depth", "visible mask", "new depth"])
        labelled = []
        for label, img in zip(labels, cells):
            cell = img.copy()
            cv2.putText(cell, f"{label} f{frame_i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cell, f"{label} f{frame_i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            labelled.append(cell)
        rows.append(np.concatenate(labelled, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phantom-native explicit IK renderer for Kinova3 + Robotiq85 + shoulders.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--setup", type=str, default="shoulders")
    parser.add_argument("--compiled-xml", type=Path, default=None)
    parser.add_argument("--background-video", type=Path, default=None)
    parser.add_argument("--source-hdf5", type=Path, default=None)
    parser.add_argument("--reference-nodepth-video", type=Path, default=None)
    parser.add_argument("--reference-depth-video", type=Path, default=None)
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--base-axis", choices=("x", "z", "minus_z"), default="minus_z")
    parser.add_argument("--root-border-pad-px", type=float, default=24.0)
    parser.add_argument("--root-depth-offset-m", type=float, default=0.10)
    parser.add_argument("--half-arm-weight", type=float, default=0.18)
    parser.add_argument("--forearm-weight", type=float, default=0.45)
    parser.add_argument("--wrist-weight", type=float, default=0.15)
    parser.add_argument("--ik-tol", type=float, default=0.035)
    parser.add_argument("--rot-tol-rad", type=float, default=0.65)
    parser.add_argument("--link-tol", type=float, default=0.10)
    parser.add_argument("--max-ik-iters", type=int, default=220)
    parser.add_argument("--depth-occlusion-margin", type=float, default=0.0)
    parser.add_argument("--determinism-label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    output_dir = (args.output_dir or processed_demo_dir / "explicit_ik_render").resolve()
    xml_path = (args.compiled_xml or output_dir / "phantom_kinova3_robotiq85_shoulders.xml").resolve()
    export_phantom_kinova_xml(xml_path)

    background, background_path = load_background(processed_demo_dir, args.background_video)
    if args.max_frames is not None:
        background = background[: args.max_frames]
    intrinsic = read_intrinsic(processed_demo_dir)
    actions = load_actions(processed_demo_dir, args.setup)
    n_frames = min(len(background), len(actions["left_pos"]), len(actions["right_pos"]))
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)
    background = background[:n_frames]
    height, width = background[0].shape[:2]
    for key in list(actions):
        actions[key] = actions[key][:n_frames]

    body_points, body_source = load_hdf5_body_points(processed_demo_dir, n_frames, args.source_hdf5)
    left_root, left_root_pixel = fixed_root_from_forearm_border(
        actions["left_pos"], body_points["left"]["Forearm"], body_points["left"]["Arm"], intrinsic, width, height, "left",
        border_pad_px=args.root_border_pad_px, depth_offset_m=args.root_depth_offset_m
    )
    right_root, right_root_pixel = fixed_root_from_forearm_border(
        actions["right_pos"], body_points["right"]["Forearm"], body_points["right"]["Arm"], intrinsic, width, height, "right",
        border_pad_px=args.root_border_pad_px, depth_offset_m=args.root_depth_offset_m
    )

    depth_path = args.depth_path
    if depth_path is None:
        candidate = processed_demo_dir / "depth_qwen_da3_clean_direct.npy"
        depth_path = candidate if candidate.exists() else None
    scene_depth = load_scene_depth(depth_path.resolve(), n_frames, height, width) if depth_path is not None and depth_path.exists() else None

    mujoco, model, data = load_model(xml_path)
    style_kinova_robot(mujoco, model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera_name = configure_camera_space_zed(mujoco, model, fovy_from_intrinsic(intrinsic, height))
    model.body_pos[body_id(mujoco, model, "robot0_base")] = right_root
    model.body_quat[body_id(mujoco, model, "robot0_base")] = base_quat(right_root, np.median(actions["right_pos"], axis=0), "right", args.base_axis)
    model.body_pos[body_id(mujoco, model, "robot1_base")] = left_root
    model.body_quat[body_id(mujoco, model, "robot1_base")] = base_quat(left_root, np.median(actions["left_pos"], axis=0), "left", args.base_axis)

    right_ee_id = body_id(mujoco, model, "gripper0_eef")
    left_ee_id = body_id(mujoco, model, "gripper1_eef")
    link_ids = {
        "right": (body_id(mujoco, model, "robot0_HalfArm2_Link"), body_id(mujoco, model, "robot0_forearm_link"), body_id(mujoco, model, "robot0_Bracelet_Link")),
        "left": (body_id(mujoco, model, "robot1_HalfArm2_Link"), body_id(mujoco, model, "robot1_forearm_link"), body_id(mujoco, model, "robot1_Bracelet_Link")),
    }

    renderer = mujoco.Renderer(model, height=height, width=width)
    set_initial_kinova_qpos(mujoco, model, data)
    previous_qpos = data.qpos.copy()
    robot_rgb: list[np.ndarray] = []
    robot_mask: list[np.ndarray] = []
    robot_depth: list[np.ndarray] = []
    nodepth_overlay: list[np.ndarray] = []
    depth_overlay: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []
    occlusion_masks: list[np.ndarray] = []
    qpos_rows = []
    metrics = []
    link_weights = (args.half_arm_weight, args.forearm_weight, args.wrist_weight)

    for frame_i in tqdm(range(n_frames), desc="Explicit IK render"):
        set_robotiq85_width(mujoco, model, data, "gripper0", float(actions["right_width"][frame_i]))
        set_robotiq85_width(mujoco, model, data, "gripper1", float(actions["left_width"][frame_i]))
        right = solve_arm_ik_selected(
            mujoco,
            model,
            data,
            target_pos=actions["right_pos"][frame_i],
            target_rot=actions["right_rot"][frame_i],
            ee_body_id=right_ee_id,
            link_targets=human_link_targets(body_points, "right", frame_i, root=right_root, half_arm_body_id=link_ids["right"][0], forearm_body_id=link_ids["right"][1], wrist_body_id=link_ids["right"][2], weights=link_weights),
            previous_qpos=previous_qpos,
            active_joint_prefixes=("robot0_Actuator",),
            pos_tol=args.ik_tol,
            rot_tol_rad=args.rot_tol_rad,
            link_tol=args.link_tol,
            max_iters=args.max_ik_iters,
        )
        left = solve_arm_ik_selected(
            mujoco,
            model,
            data,
            target_pos=actions["left_pos"][frame_i],
            target_rot=actions["left_rot"][frame_i],
            ee_body_id=left_ee_id,
            link_targets=human_link_targets(body_points, "left", frame_i, root=left_root, half_arm_body_id=link_ids["left"][0], forearm_body_id=link_ids["left"][1], wrist_body_id=link_ids["left"][2], weights=link_weights),
            previous_qpos=right.qpos,
            active_joint_prefixes=("robot1_Actuator",),
            pos_tol=args.ik_tol,
            rot_tol_rad=args.rot_tol_rad,
            link_tol=args.link_tol,
            max_iters=args.max_ik_iters,
        )
        data.qpos[:] = left.qpos
        set_robotiq85_width(mujoco, model, data, "gripper0", float(actions["right_width"][frame_i]))
        set_robotiq85_width(mujoco, model, data, "gripper1", float(actions["left_width"][frame_i]))
        mujoco.mj_forward(model, data)
        previous_qpos = data.qpos.copy()
        rgb, mask, rdepth = render_rgb_mask_depth(mujoco, renderer, model, data, camera_name)
        robot_rgb.append(rgb)
        robot_mask.append(mask)
        robot_depth.append(rdepth)
        qpos_rows.append(data.qpos.copy())
        nodepth_overlay.append(overlay_robot(background[frame_i], rgb, mask))
        if scene_depth is not None:
            overlay, visible, occluded = overlay_with_depth(background[frame_i], rgb, mask, rdepth, scene_depth[frame_i], args.depth_occlusion_margin)
            depth_overlay.append(overlay)
            visible_masks.append(visible)
            occlusion_masks.append(occluded)
        metrics.append(
            {
                "frame": frame_i,
                "left_width": float(actions["left_width"][frame_i]),
                "right_width": float(actions["right_width"][frame_i]),
                "left_pos_err_m": left.pos_err_m,
                "right_pos_err_m": right.pos_err_m,
                "left_rot_err_rad": left.rot_err_rad,
                "right_rot_err_rad": right.rot_err_rad,
                "left_link_err_m": left.link_err_m,
                "right_link_err_m": right.link_err_m,
                "left_seed_id": left.seed_id,
                "right_seed_id": right.seed_id,
                "left_converged": left.converged,
                "right_converged": right.converged,
                "qpos_delta_norm": float(np.linalg.norm(data.qpos - qpos_rows[-2])) if len(qpos_rows) > 1 else 0.0,
                "robot_mask_area": float(mask.mean()),
                "visible_mask_area": float(visible_masks[-1].mean()) if visible_masks else None,
            }
        )
    renderer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    robot_rgb_arr = np.asarray(robot_rgb, dtype=np.uint8)
    robot_mask_arr = np.asarray(robot_mask, dtype=bool)
    robot_depth_arr = np.asarray(robot_depth, dtype=np.float32)
    visible_arr = np.asarray(visible_masks, dtype=bool) if visible_masks else np.zeros((0, height, width), dtype=bool)
    occlusion_arr = np.asarray(occlusion_masks, dtype=bool) if occlusion_masks else np.zeros((0, height, width), dtype=bool)
    qpos_arr = np.asarray(qpos_rows, dtype=np.float32)
    np.savez_compressed(
        output_dir / "robot_render_canonical.npz",
        qpos=qpos_arr,
        robot_rgb=robot_rgb_arr,
        robot_mask=robot_mask_arr,
        robot_depth=robot_depth_arr,
        visible_mask=visible_arr,
        occlusion_mask=occlusion_arr,
        left_target_width=actions["left_width"].astype(np.float32),
        right_target_width=actions["right_width"].astype(np.float32),
    )
    save_video(robot_rgb, output_dir / "video_robot_Kinova3_shoulders_explicit_ik.mp4", args.fps)
    save_video(nodepth_overlay, output_dir / "video_overlay_Kinova3_shoulders_explicit_ik_nodepth.mp4", args.fps)
    if depth_overlay:
        save_video(depth_overlay, output_dir / "video_overlay_Kinova3_shoulders_explicit_ik_depth.mp4", args.fps)
    montage_frames = [i for i in KEY_FRAMES if i < n_frames]
    ref_nodepth_path = args.reference_nodepth_video
    if ref_nodepth_path is None:
        candidate = processed_demo_dir / "video_overlay_Kinova3_shoulders_qwen_inpaint_sam3_d5.mp4"
        ref_nodepth_path = candidate if candidate.exists() else None
    ref_depth_path = args.reference_depth_video
    if ref_depth_path is None:
        candidate = processed_demo_dir / "video_overlay_Kinova3_shoulders_qwen_inpaint_sam3_d5_da3depth_direct.mp4"
        ref_depth_path = candidate if candidate.exists() else None
    reference_nodepth = read_video_frames(ref_nodepth_path)[:n_frames] if ref_nodepth_path is not None and ref_nodepth_path.exists() else None
    reference_depth = read_video_frames(ref_depth_path)[:n_frames] if ref_depth_path is not None and ref_depth_path.exists() else None
    make_compare_montage(
        output_dir / "explicit_ik_compare_montage.jpg",
        montage_frames,
        background,
        reference_nodepth,
        reference_depth,
        robot_rgb_arr,
        nodepth_overlay,
        scene_depth,
        visible_arr if len(visible_arr) else np.zeros((n_frames, height, width), dtype=bool),
        depth_overlay if depth_overlay else None,
    )
    key_metrics = {str(row["frame"]): row for row in metrics if row["frame"] in montage_frames}
    summary = {
        "stage": "phantom_native_explicit_ik_render",
        "processed_demo_dir": str(processed_demo_dir),
        "output_dir": str(output_dir),
        "compiled_xml": str(xml_path),
        "background_video": background_path,
        "depth_path": str(depth_path.resolve()) if depth_path is not None and depth_path.exists() else None,
        "reference_nodepth_video": str(ref_nodepth_path.resolve()) if ref_nodepth_path is not None and ref_nodepth_path.exists() else None,
        "reference_depth_video": str(ref_depth_path.resolve()) if ref_depth_path is not None and ref_depth_path.exists() else None,
        "depth_occlusion_margin": float(args.depth_occlusion_margin),
        "frames": int(n_frames),
        "fps": float(args.fps),
        "resolution": [int(width), int(height)],
        "body_points_source": body_source,
        "left_root_camera": left_root.tolist(),
        "right_root_camera": right_root.tolist(),
        "left_root_pixel": list(left_root_pixel),
        "right_root_pixel": list(right_root_pixel),
        "mean_left_pos_err_m": float(np.mean([m["left_pos_err_m"] for m in metrics])),
        "mean_right_pos_err_m": float(np.mean([m["right_pos_err_m"] for m in metrics])),
        "max_left_pos_err_m": float(np.max([m["left_pos_err_m"] for m in metrics])),
        "max_right_pos_err_m": float(np.max([m["right_pos_err_m"] for m in metrics])),
        "left_converged_ratio": float(np.mean([m["left_converged"] for m in metrics])),
        "right_converged_ratio": float(np.mean([m["right_converged"] for m in metrics])),
        "mean_robot_mask_area": float(robot_mask_arr.mean()),
        "mean_visible_mask_area": float(visible_arr.mean()) if len(visible_arr) else None,
        "key_frame_metrics": key_metrics,
        "canonical_npz": str(output_dir / "robot_render_canonical.npz"),
        "robot_video": str(output_dir / "video_robot_Kinova3_shoulders_explicit_ik.mp4"),
        "nodepth_overlay": str(output_dir / "video_overlay_Kinova3_shoulders_explicit_ik_nodepth.mp4"),
        "depth_overlay": str(output_dir / "video_overlay_Kinova3_shoulders_explicit_ik_depth.mp4") if depth_overlay else None,
        "montage": str(output_dir / "explicit_ik_compare_montage.jpg"),
        "determinism_label": args.determinism_label,
    }
    write_json(output_dir / "explicit_ik_summary.json", summary)
    print(output_dir / "explicit_ik_summary.json")


if __name__ == "__main__":
    main()
