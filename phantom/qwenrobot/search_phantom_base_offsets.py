from __future__ import annotations

import argparse
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.qwenrobot.gripper_frame_utils import apply_frame_candidate, recover_qwen_rotations_from_legacy
from phantom.qwenrobot.prepare_egodex_for_phantom import add_phantom_submodules_to_path, build_phantom_cfg
from phantom.qwenrobot.rerender_robot_overlay import find_intrinsics, output_root_from_processed

PHANTOM_TOOL_OFFSET = Rotation.from_euler("z", 135.0, degrees=True).as_matrix()
QWEN_TO_PHANTOM_TOOL = (
    Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    @ Rotation.from_euler("z", -135.0, degrees=True).as_matrix()
)
OPENING_AXIS_LOCAL = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_TARGET_CENTER = np.asarray([456.0 * 0.5, 256.0 * 0.5], dtype=np.float64)


def representative_keyframes(left_xyz: np.ndarray, right_xyz: np.ndarray) -> np.ndarray:
    ids: set[int] = {0, len(left_xyz) - 1, len(left_xyz) // 2}
    for xyz in (left_xyz, right_xyz):
        for dim in range(3):
            ids.add(int(np.argmin(xyz[:, dim])))
            ids.add(int(np.argmax(xyz[:, dim])))
    return np.asarray(sorted(idx for idx in ids if 0 <= idx < len(left_xyz)), dtype=np.int64)


def candidate_offsets(radius: float, step: float, z_offsets: list[float]) -> list[np.ndarray]:
    values = np.arange(-radius, radius + 1e-6, step, dtype=np.float64)
    rows = []
    for dx in values:
        for dy in values:
            for dz in z_offsets:
                rows.append(np.asarray([dx, dy, dz], dtype=np.float64))
    return rows


def parse_candidate_offsets(values: list[str] | list[list[str]] | None) -> list[np.ndarray] | None:
    if not values:
        return None
    if values and isinstance(values[0], list):
        values = [item for group in values for item in group]
    offsets = []
    for value in values:
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--candidate-offsets entries must be x,y,z, got {value!r}")
        offsets.append(np.asarray(parts, dtype=np.float64))
    return offsets


def reject_negative_z_offsets(offsets: list[np.ndarray], name: str, allow_negative_z: bool) -> None:
    if allow_negative_z:
        return
    bad = [offset.tolist() for offset in offsets if float(offset[2]) < -1e-9]
    if bad:
        raise ValueError(
            f"{name} contains negative z offsets, which are disabled to prevent ground/table embedding: {bad[:5]}. "
            "Use --allow-negative-z-offsets only for explicit diagnostics."
        )


def candidate_configs(
    offset_pairs: list[tuple[np.ndarray, np.ndarray]],
    yaw_degs: list[float],
    tool_roll_degs: list[float],
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    rows = []
    for base0_offset, base1_offset in offset_pairs:
        for yaw_deg in yaw_degs:
            for tool_roll_deg in tool_roll_degs:
                rows.append((base0_offset, base1_offset, float(yaw_deg), float(tool_roll_deg)))
    return rows


def make_offset_pairs(
    shared_offsets: list[np.ndarray],
    base0_offsets: list[np.ndarray] | None,
    base1_offsets: list[np.ndarray] | None,
    independent: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if base0_offsets is None:
        base0_offsets = shared_offsets
    if base1_offsets is None:
        base1_offsets = shared_offsets
    if independent or base0_offsets is not shared_offsets or base1_offsets is not shared_offsets:
        return [(base0, base1) for base0 in base0_offsets for base1 in base1_offsets]
    return [(offset, offset) for offset in shared_offsets]


def set_base_env(base0_offset: np.ndarray, base1_offset: np.ndarray, yaw_deg: float) -> None:
    os.environ["PHANTOM_BIMANUAL_BASE0_OFFSET"] = ",".join(f"{float(v):.6g}" for v in base0_offset)
    os.environ["PHANTOM_BIMANUAL_BASE1_OFFSET"] = ",".join(f"{float(v):.6g}" for v in base1_offset)
    os.environ["PHANTOM_BIMANUAL_GLOBAL_YAW_DEG"] = f"{float(yaw_deg):.6g}"


def clear_base_env() -> None:
    os.environ.pop("PHANTOM_BIMANUAL_BASE0_OFFSET", None)
    os.environ.pop("PHANTOM_BIMANUAL_BASE1_OFFSET", None)
    os.environ.pop("PHANTOM_BIMANUAL_GLOBAL_YAW_DEG", None)


def load_smoothed(processed_demo_dir: Path, setup: str) -> dict[str, np.ndarray]:
    smoothing_dir = processed_demo_dir / "smoothing_processor"
    data = {}
    for side in ("left", "right"):
        path = smoothing_dir / f"smoothed_actions_{side}_{setup}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path)
        data[f"{side}_pos"] = arr["ee_pts"].astype(np.float64)
        data[f"{side}_rot"] = arr["ee_oris"].astype(np.float64)
        data[f"{side}_width"] = arr["ee_widths"].astype(np.float64)
    return data


def load_hand_centers(processed_demo_dir: Path, setup: str) -> dict[str, np.ndarray] | None:
    hand_dir = processed_demo_dir / "hand_processor"
    action_path = processed_demo_dir / "action_processor" / f"actions_left_{setup}.npz"
    left_path = hand_dir / "hand_data_left.npz"
    right_path = hand_dir / "hand_data_right.npz"
    if not (action_path.exists() and left_path.exists() and right_path.exists()):
        return None
    with np.load(action_path, allow_pickle=True) as action_data:
        union_indices = np.asarray(action_data["union_indices"], dtype=np.int64)
    centers = {}
    for side, path in (("left", left_path), ("right", right_path)):
        with np.load(path) as data:
            kpts = np.asarray(data["kpts_2d"], dtype=np.float64)
            detected = np.asarray(data["hand_detected"], dtype=bool)
        side_centers = np.full((len(union_indices), 2), np.nan, dtype=np.float64)
        for local_idx, raw_idx in enumerate(union_indices):
            if raw_idx < 0 or raw_idx >= len(kpts) or not detected[raw_idx]:
                continue
            pts = kpts[raw_idx]
            valid = np.isfinite(pts).all(axis=1)
            if np.any(valid):
                side_centers[local_idx] = pts[valid].mean(axis=0)
        centers[side] = side_centers
    return centers


def apply_qwen_tool_axis_fix_in_memory(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fixed = dict(traj)
    for side in ("left", "right"):
        fixed[f"{side}_rot"] = np.asarray(traj[f"{side}_rot"]) @ QWEN_TO_PHANTOM_TOOL
    return fixed


def apply_tool_roll(rot: np.ndarray, tool_roll_deg: float) -> np.ndarray:
    if abs(tool_roll_deg) < 1e-12:
        return rot
    return rot @ Rotation.from_euler("x", tool_roll_deg, degrees=True).as_matrix()


def parse_gripper_frame_matrix(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != 9:
        raise ValueError("--gripper-frame-matrix must contain 9 floats in row-major order")
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"--gripper-frame-matrix must be a proper rotation, det={det}")
    return matrix


def apply_gripper_frame_calibration(
    traj: dict[str, np.ndarray],
    *,
    recover_qwen_from_legacy: bool,
    gripper_frame_matrix: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if not recover_qwen_from_legacy and gripper_frame_matrix is None:
        return traj
    fixed = dict(traj)
    for side in ("left", "right"):
        rotations = np.asarray(traj[f"{side}_rot"], dtype=np.float64)
        if recover_qwen_from_legacy:
            rotations = recover_qwen_rotations_from_legacy(rotations)
        if gripper_frame_matrix is not None:
            rotations = rotations @ gripper_frame_matrix
        fixed[f"{side}_rot"] = rotations
    return fixed


def controller_target_rot(rot: np.ndarray) -> np.ndarray:
    return rot @ PHANTOM_TOOL_OFFSET


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return float("inf")
    cos = float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos)))


def rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(target.T @ actual).magnitude()))


def visual_metrics(robot_mask: np.ndarray, gripper_mask: np.ndarray) -> dict[str, float]:
    robot_mask = np.squeeze(robot_mask).astype(bool)
    gripper_mask = np.squeeze(gripper_mask).astype(bool)
    full_mask = robot_mask | gripper_mask
    if full_mask.ndim != 2:
        raise ValueError(f"Expected 2D render mask, got {full_mask.shape}")
    h = full_mask.shape[0]
    w = full_mask.shape[1]
    bottom = full_mask[int(h * 0.67):]
    if np.any(full_mask):
        ys, xs = np.nonzero(full_mask)
        bbox_area_ratio = float(((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / (h * w))
        bbox_bottom_touch = float(ys.max() >= int(0.95 * h))
        bbox_center_y = float(((ys.min() + ys.max()) * 0.5) / h)
    else:
        bbox_area_ratio = 0.0
        bbox_bottom_touch = 0.0
        bbox_center_y = 0.0
    return {
        "robot_area_ratio": float(full_mask.mean()),
        "gripper_area_ratio": float(gripper_mask.mean()),
        "bottom_area_ratio": float(bottom.mean()) if bottom.size else 0.0,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_bottom_touch": bbox_bottom_touch,
        "bbox_center_y": bbox_center_y,
    }


def mask_centroid_xy(mask: np.ndarray) -> np.ndarray | None:
    mask = np.squeeze(mask).astype(bool)
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def safe_nanmean(values: list[float], default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float(default)
    return float(np.nanmean(arr))


def side_projection_metrics(
    result: dict[str, np.ndarray],
    hand_centers: dict[str, np.ndarray] | None,
    frame_idx: int,
) -> dict[str, float]:
    if hand_centers is None:
        return {
            "right_projection_error_px": float("nan"),
            "left_projection_error_px": float("nan"),
            "mean_projection_error_px": float("nan"),
            "crossing_penalty_px": 0.0,
        }
    right_center = hand_centers["right"][frame_idx]
    left_center = hand_centers["left"][frame_idx]
    right_robot = mask_centroid_xy(result.get("right_gripper_mask", result["gripper_mask"]))
    left_robot = mask_centroid_xy(result.get("left_gripper_mask", result["gripper_mask"]))
    if right_robot is None:
        right_robot = mask_centroid_xy(result.get("right_robot_mask", result["robot_mask"]))
    if left_robot is None:
        left_robot = mask_centroid_xy(result.get("left_robot_mask", result["robot_mask"]))
    right_err = float("nan")
    left_err = float("nan")
    if right_robot is not None and np.isfinite(right_center).all():
        right_err = float(np.linalg.norm(right_robot - right_center))
    if left_robot is not None and np.isfinite(left_center).all():
        left_err = float(np.linalg.norm(left_robot - left_center))
    finite_errors = [value for value in (right_err, left_err) if np.isfinite(value)]
    mean_err = float(np.mean(finite_errors)) if finite_errors else float("nan")
    crossing_penalty = 0.0
    if (
        right_robot is not None
        and left_robot is not None
        and np.isfinite(right_center).all()
        and np.isfinite(left_center).all()
    ):
        target_dx = float(right_center[0] - left_center[0])
        robot_dx = float(right_robot[0] - left_robot[0])
        if abs(target_dx) > 10.0 and target_dx * robot_dx < 0.0:
            crossing_penalty = abs(target_dx) + abs(robot_dx)
    return {
        "right_projection_error_px": right_err,
        "left_projection_error_px": left_err,
        "mean_projection_error_px": mean_err,
        "crossing_penalty_px": float(crossing_penalty),
    }


def make_cfg(processed_demo_dir: Path, extrinsics: Path, robot: str, gripper: str, setup: str, input_resolution: int, output_resolution: int):
    output_root, demo_name, demo_num = output_root_from_processed(processed_demo_dir)
    intrinsics = find_intrinsics(output_root, demo_name, demo_num)
    cfg_args = Namespace(
        output_root=output_root,
        demo_name=demo_name,
        demo_num=demo_num,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        robot=robot,
        gripper=gripper,
        bimanual_setup=setup,
        depth_for_overlay=False,
        depth_occlusion_margin=0.03,
    )
    return build_phantom_cfg(cfg_args, intrinsics, extrinsics), demo_num


def evaluate_candidate(
    cfg,
    demo_num: str,
    traj: dict[str, np.ndarray],
    keyframes: np.ndarray,
    base0_offset: np.ndarray,
    base1_offset: np.ndarray,
    yaw_deg: float,
    tool_roll_deg: float,
    tracking_threshold: float,
    orientation_mode: str,
    orientation_threshold_deg: float,
    hand_centers: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

    set_base_env(base0_offset, base1_offset, yaw_deg)
    processor = None
    try:
        processor = RobotInpaintProcessor(cfg)
        valid = []
        errors = []
        axis_errors = []
        rotation_errors = []
        robot_areas = []
        gripper_areas = []
        bottom_areas = []
        bbox_areas = []
        bbox_bottom_touches = []
        bbox_center_ys = []
        right_projection_errors = []
        left_projection_errors = []
        mean_projection_errors = []
        crossing_penalties = []
        for local_idx, frame_idx in enumerate(keyframes):
            right_rot = apply_tool_roll(traj["right_rot"][frame_idx], tool_roll_deg)
            left_rot = apply_tool_roll(traj["left_rot"][frame_idx], tool_roll_deg)
            target_state = {
                "pos": [traj["right_pos"][frame_idx], traj["left_pos"][frame_idx]],
                "ori_xyzw": [
                    Rotation.from_matrix(right_rot).as_quat(scalar_first=False),
                    Rotation.from_matrix(left_rot).as_quat(scalar_first=False),
                ],
                "gripper_pos": [traj["right_width"][frame_idx], traj["left_width"][frame_idx]],
            }
            result = processor.twin_robot.move_to_target_state(target_state, init=(local_idx == 0))
            err = max(float(result["left_pos_err"]), float(result["right_pos_err"]))
            right_target_rot = controller_target_rot(right_rot)
            left_target_rot = controller_target_rot(left_rot)
            right_axis_err = angle_between_deg(result["right_eef_ori"] @ OPENING_AXIS_LOCAL, right_target_rot @ OPENING_AXIS_LOCAL)
            left_axis_err = angle_between_deg(result["left_eef_ori"] @ OPENING_AXIS_LOCAL, left_target_rot @ OPENING_AXIS_LOCAL)
            right_rot_err = rotation_error_deg(result["right_eef_ori"], right_target_rot)
            left_rot_err = rotation_error_deg(result["left_eef_ori"], left_target_rot)
            visual = visual_metrics(result["robot_mask"], result["gripper_mask"])
            projection = side_projection_metrics(result, hand_centers, int(frame_idx))
            axis_err = max(right_axis_err, left_axis_err)
            rot_err = max(right_rot_err, left_rot_err)
            errors.append(err)
            axis_errors.append(axis_err)
            rotation_errors.append(rot_err)
            robot_areas.append(visual["robot_area_ratio"])
            gripper_areas.append(visual["gripper_area_ratio"])
            bottom_areas.append(visual["bottom_area_ratio"])
            bbox_areas.append(visual["bbox_area_ratio"])
            bbox_bottom_touches.append(visual["bbox_bottom_touch"])
            bbox_center_ys.append(visual["bbox_center_y"])
            right_projection_errors.append(projection["right_projection_error_px"])
            left_projection_errors.append(projection["left_projection_error_px"])
            mean_projection_errors.append(projection["mean_projection_error_px"])
            crossing_penalties.append(projection["crossing_penalty_px"])
            if orientation_mode == "qwen-exact":
                valid.append(err <= tracking_threshold and rot_err <= orientation_threshold_deg)
            else:
                valid.append(err <= tracking_threshold)
        return {
            "offset": base0_offset.tolist() if np.allclose(base0_offset, base1_offset) else None,
            "base0_offset": base0_offset.tolist(),
            "base1_offset": base1_offset.tolist(),
            "global_yaw_deg": float(yaw_deg),
            "tool_roll_deg": float(tool_roll_deg),
            "orientation_mode": orientation_mode,
            "valid_ratio": float(np.mean(valid)),
            "valid_keyframes": int(np.sum(valid)),
            "total_keyframes": int(len(keyframes)),
            "mean_tracking_error": float(np.mean(errors)),
            "max_tracking_error": float(np.max(errors)),
            "mean_axis_error_deg": float(np.mean(axis_errors)),
            "max_axis_error_deg": float(np.max(axis_errors)),
            "mean_rotation_error_deg": float(np.mean(rotation_errors)),
            "max_rotation_error_deg": float(np.max(rotation_errors)),
            "mean_robot_area_ratio": float(np.mean(robot_areas)),
            "max_robot_area_ratio": float(np.max(robot_areas)),
            "mean_gripper_area_ratio": float(np.mean(gripper_areas)),
            "mean_bottom_area_ratio": float(np.mean(bottom_areas)),
            "max_bottom_area_ratio": float(np.max(bottom_areas)),
            "mean_bbox_area_ratio": float(np.mean(bbox_areas)),
            "mean_bbox_bottom_touch": float(np.mean(bbox_bottom_touches)),
            "mean_bbox_center_y": float(np.mean(bbox_center_ys)),
            "mean_right_projection_error_px": safe_nanmean(right_projection_errors),
            "mean_left_projection_error_px": safe_nanmean(left_projection_errors),
            "mean_projection_error_px": safe_nanmean(mean_projection_errors),
            "mean_crossing_penalty_px": safe_nanmean(crossing_penalties, default=0.0),
        }
    finally:
        if processor is not None:
            processor.__del__()
        clear_base_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Phantom shoulders base offsets for a processed demo.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--camera-extrinsics", type=Path, required=True)
    parser.add_argument("--robot", type=str, default="Kinova3")
    parser.add_argument("--gripper", type=str, default="Robotiq85")
    parser.add_argument("--bimanual-setup", type=str, default="shoulders")
    parser.add_argument("--input-resolution", type=int, default=512)
    parser.add_argument("--output-resolution", type=int, default=512)
    parser.add_argument("--xy-radius", type=float, default=0.30)
    parser.add_argument("--xy-step", type=float, default=0.10)
    parser.add_argument("--z-offsets", type=float, nargs="*", default=[0.0, 0.10])
    parser.add_argument("--candidate-offsets", type=str, nargs="+", action="append", default=None)
    parser.add_argument("--candidate-base0-offsets", type=str, nargs="+", action="append", default=None)
    parser.add_argument("--candidate-base1-offsets", type=str, nargs="+", action="append", default=None)
    parser.add_argument("--independent-base-offsets", action="store_true")
    parser.add_argument("--allow-negative-z-offsets", action="store_true")
    parser.add_argument("--yaw-degs", type=float, nargs="*", default=[0.0, 15.0, -15.0, 30.0, -30.0])
    parser.add_argument("--tool-roll-degs", type=float, nargs="*", default=[0.0, 45.0, -45.0, 90.0, -90.0])
    parser.add_argument("--orientation-mode", choices=("jaw-axis-soft", "qwen-exact"), default="jaw-axis-soft")
    parser.add_argument("--orientation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--tracking-threshold", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--search-strategy", choices=("staged", "grid"), default="staged")
    parser.add_argument("--top-offsets", type=int, default=10)
    parser.add_argument("--visual-aware-ranking", action="store_true")
    parser.add_argument("--min-valid-ratio-for-visual-sort", type=float, default=0.80)
    parser.add_argument("--visual-area-weight", type=float, default=0.50)
    parser.add_argument("--bottom-area-weight", type=float, default=0.50)
    parser.add_argument("--bbox-bottom-weight", type=float, default=0.20)
    parser.add_argument("--projection-error-weight", type=float, default=0.002)
    parser.add_argument("--crossing-penalty-weight", type=float, default=0.004)
    parser.add_argument("--max-bottom-area-for-visual-sort", type=float, default=None)
    parser.add_argument("--max-bbox-bottom-touch-for-visual-sort", type=float, default=None)
    parser.add_argument("--max-projection-error-for-visual-sort", type=float, default=None)
    parser.add_argument("--recover-qwen-from-legacy", action="store_true")
    parser.add_argument("--gripper-frame-matrix", type=float, nargs=9, default=None)
    parser.add_argument("--gripper-frame-name", type=str, default=None)
    parser.add_argument("--qwen-tool-axis-fix", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "WARNING: search_phantom_base_offsets is legacy Phantom/robosuite shoulders-offset search. "
        "Use `python -m phantom.qwenrobot.search_base` for the direct MuJoCo xyz+yaw base search.",
        flush=True,
    )
    add_phantom_submodules_to_path()
    processed_demo_dir = args.processed_demo_dir.resolve()
    traj = load_smoothed(processed_demo_dir, args.bimanual_setup)
    if args.qwen_tool_axis_fix:
        traj = apply_qwen_tool_axis_fix_in_memory(traj)
    gripper_frame_matrix = parse_gripper_frame_matrix(args.gripper_frame_matrix)
    traj = apply_gripper_frame_calibration(
        traj,
        recover_qwen_from_legacy=bool(args.recover_qwen_from_legacy),
        gripper_frame_matrix=gripper_frame_matrix,
    )
    keyframes = representative_keyframes(traj["left_pos"], traj["right_pos"])
    hand_centers = load_hand_centers(processed_demo_dir, args.bimanual_setup)
    cfg, demo_num = make_cfg(
        processed_demo_dir,
        args.camera_extrinsics.resolve(),
        args.robot,
        args.gripper,
        args.bimanual_setup,
        args.input_resolution,
        args.output_resolution,
    )
    shared_offsets = parse_candidate_offsets(args.candidate_offsets)
    if shared_offsets is None:
        shared_offsets = candidate_offsets(args.xy_radius, args.xy_step, args.z_offsets)
    base0_offsets = parse_candidate_offsets(args.candidate_base0_offsets)
    base1_offsets = parse_candidate_offsets(args.candidate_base1_offsets)
    reject_negative_z_offsets(shared_offsets, "--candidate-offsets/--z-offsets", args.allow_negative_z_offsets)
    if base0_offsets is not None:
        reject_negative_z_offsets(base0_offsets, "--candidate-base0-offsets", args.allow_negative_z_offsets)
    if base1_offsets is not None:
        reject_negative_z_offsets(base1_offsets, "--candidate-base1-offsets", args.allow_negative_z_offsets)
    offset_pairs = make_offset_pairs(
        shared_offsets,
        base0_offsets,
        base1_offsets,
        args.independent_base_offsets,
    )
    if args.max_candidates is not None:
        offset_pairs = offset_pairs[: args.max_candidates]

    stage1_rows = []
    if args.search_strategy == "staged":
        for base0_offset, base1_offset in offset_pairs:
            row = evaluate_candidate(
                cfg,
                demo_num,
                traj,
                keyframes,
                base0_offset,
                base1_offset,
                0.0,
                0.0,
                args.tracking_threshold,
                args.orientation_mode,
                args.orientation_threshold_deg,
                hand_centers,
            )
            stage1_rows.append(row)
            print(json.dumps({"stage": "offset", **row}), flush=True)
        ranked_offsets = sorted(
            stage1_rows,
            key=lambda row: (
                row["valid_ratio"],
                -row["mean_tracking_error"],
                -row["mean_axis_error_deg"],
                -row["max_tracking_error"],
            ),
            reverse=True,
        )
        offset_pairs = [
            (
                np.asarray(row["base0_offset"], dtype=np.float64),
                np.asarray(row["base1_offset"], dtype=np.float64),
            )
            for row in ranked_offsets[: args.top_offsets]
        ]

    candidates = candidate_configs(offset_pairs, args.yaw_degs, args.tool_roll_degs)
    rows = []
    for base0_offset, base1_offset, yaw_deg, tool_roll_deg in candidates:
        rows.append(
            evaluate_candidate(
                cfg,
                demo_num,
                traj,
                keyframes,
                base0_offset,
                base1_offset,
                yaw_deg,
                tool_roll_deg,
                args.tracking_threshold,
                args.orientation_mode,
                args.orientation_threshold_deg,
                hand_centers,
            )
        )
        print(json.dumps({"stage": "yaw_tool", **rows[-1]}), flush=True)
    def visual_score(row: dict[str, object]) -> float:
        return (
            float(row["mean_tracking_error"])
            + args.visual_area_weight * float(row.get("mean_robot_area_ratio", 0.0))
            + args.bottom_area_weight * float(row.get("mean_bottom_area_ratio", 0.0))
            + args.bbox_bottom_weight * float(row.get("mean_bbox_bottom_touch", 0.0))
            + args.projection_error_weight * (
                0.0
                if not np.isfinite(float(row.get("mean_projection_error_px", float("nan"))))
                else float(row.get("mean_projection_error_px", 0.0))
            )
            + args.crossing_penalty_weight * float(row.get("mean_crossing_penalty_px", 0.0))
        )

    for row in rows:
        row["visual_score"] = float(visual_score(row))

    if args.visual_aware_ranking:
        eligible = [
            row for row in rows
            if float(row["valid_ratio"]) >= args.min_valid_ratio_for_visual_sort
            and (
                args.max_bottom_area_for_visual_sort is None
                or float(row.get("mean_bottom_area_ratio", 0.0)) <= args.max_bottom_area_for_visual_sort
            )
            and (
                args.max_bbox_bottom_touch_for_visual_sort is None
                or float(row.get("mean_bbox_bottom_touch", 0.0)) <= args.max_bbox_bottom_touch_for_visual_sort
            )
            and (
                args.max_projection_error_for_visual_sort is None
                or not np.isfinite(float(row.get("mean_projection_error_px", float("nan"))))
                or float(row.get("mean_projection_error_px", 0.0)) <= args.max_projection_error_for_visual_sort
            )
        ]
        pool = eligible if eligible else rows
        best = min(
            pool,
            key=lambda row: (
                float(row["visual_score"]),
                -float(row["valid_ratio"]),
                float(row["mean_tracking_error"]),
                float(row["mean_axis_error_deg"]),
            ),
        )
    else:
        best = max(
            rows,
            key=lambda row: (
                row["valid_ratio"],
                -row["mean_tracking_error"],
                -row["mean_axis_error_deg"],
                -row["max_tracking_error"],
            ),
        )
    payload = {
        "processed_demo_dir": str(processed_demo_dir),
        "robot": args.robot,
        "gripper": args.gripper,
        "bimanual_setup": args.bimanual_setup,
        "keyframes": keyframes.tolist(),
        "num_candidates": len(rows),
        "num_stage1_candidates": len(stage1_rows),
        "search_strategy": args.search_strategy,
        "top_offsets": int(args.top_offsets),
        "tracking_threshold": float(args.tracking_threshold),
        "orientation_mode": args.orientation_mode,
        "orientation_threshold_deg": float(args.orientation_threshold_deg),
        "independent_base_offsets": bool(args.independent_base_offsets),
        "allow_negative_z_offsets": bool(args.allow_negative_z_offsets),
        "qwen_tool_axis_fix": bool(args.qwen_tool_axis_fix),
        "recover_qwen_from_legacy": bool(args.recover_qwen_from_legacy),
        "gripper_frame_name": args.gripper_frame_name,
        "gripper_frame_matrix": gripper_frame_matrix.tolist() if gripper_frame_matrix is not None else None,
        "visual_aware_ranking": bool(args.visual_aware_ranking),
        "min_valid_ratio_for_visual_sort": float(args.min_valid_ratio_for_visual_sort),
        "visual_area_weight": float(args.visual_area_weight),
        "bottom_area_weight": float(args.bottom_area_weight),
        "bbox_bottom_weight": float(args.bbox_bottom_weight),
        "projection_error_weight": float(args.projection_error_weight),
        "crossing_penalty_weight": float(args.crossing_penalty_weight),
        "max_bottom_area_for_visual_sort": args.max_bottom_area_for_visual_sort,
        "max_bbox_bottom_touch_for_visual_sort": args.max_bbox_bottom_touch_for_visual_sort,
        "max_projection_error_for_visual_sort": args.max_projection_error_for_visual_sort,
        "hand_projection_centers_available": hand_centers is not None,
        "yaw_search_supported": True,
        "global_yaw_deg": best["global_yaw_deg"],
        "tool_roll_deg": best["tool_roll_deg"],
        "base0_offset": best["base0_offset"],
        "base1_offset": best["base1_offset"],
        "best": best,
        "candidates": rows,
    }
    output = args.output.resolve() if args.output else processed_demo_dir / "base_search" / "best_base.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
