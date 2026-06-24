from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

KEY_FRAMES = (27, 28, 29, 30, 31, 52, 75, 98)
INVALID_MARKER = "INVALID_NO_HDF5.json"


class InputValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalInputs:
    processed_demo_dir: Path
    trajectory_npz: Path
    output_dir: Path
    source_hdf5: Path
    source_hdf5_origin: str
    background_video: Path


def _scalar_string(arr: np.ndarray) -> str | None:
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()
    else:
        return None
    if value is None:
        return None
    text = str(value)
    return text if text else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def save_video(frames: list[np.ndarray], output_video: Path, fps: float) -> None:
    import imageio.v2 as imageio

    output_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_video), frames, fps=float(fps), quality=8, macro_block_size=1)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def read_video_frames(path: Path) -> list[np.ndarray]:
    import cv2

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


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def ray_to_padded_image_border(anchor: np.ndarray, direction: np.ndarray, width: int, height: int, pad: float) -> np.ndarray:
    direction = normalize(direction[:2])
    x0, y0 = float(anchor[0]), float(anchor[1])
    candidates: list[tuple[float, np.ndarray]] = []
    min_x, max_x, min_y, max_y = (-pad, float(width) + pad, -pad, float(height) + pad)
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


def base_quat(root: np.ndarray, target: np.ndarray, side: str, axis: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

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


def resolve_source_hdf5(
    *,
    explicit_source_hdf5: Path | None,
    processed_demo_dir: Path,
    trajectory_npz: Path,
) -> tuple[Path, str]:
    if explicit_source_hdf5 is not None:
        return explicit_source_hdf5.expanduser().resolve(), "cli"

    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = str(manifest.get("source_hdf5") or "")
        if value:
            return Path(value).expanduser().resolve(), "adapter_manifest"

    with np.load(trajectory_npz, allow_pickle=False) as traj:
        if "source_hdf5" in traj.files:
            value = _scalar_string(traj["source_hdf5"])
            if value:
                return Path(value).expanduser().resolve(), "trajectory_npz"

    raise InputValidationError(
        "No source HDF5 was provided. Use --source-hdf5, adapter_manifest.json source_hdf5, "
        "or a source_hdf5 scalar in --trajectory-npz."
    )


def validate_hdf5_source(hdf5_path: Path) -> None:
    required = [
        "transforms/camera",
        "transforms/leftForearm",
        "transforms/leftHand",
        "transforms/leftArm",
        "transforms/rightForearm",
        "transforms/rightHand",
        "transforms/rightArm",
    ]
    try:
        if not hdf5_path.exists():
            raise FileNotFoundError(f"source_hdf5 not accessible: {hdf5_path}")
        with h5py.File(str(hdf5_path), "r") as h5:
            missing = [key for key in required if key not in h5]
            if missing:
                raise KeyError(f"source_hdf5 missing required datasets: {missing}")
            for key in required:
                _ = h5[key].shape
    except Exception as exc:
        raise InputValidationError(
            "Cannot run canonical render without readable EgoDex HDF5 body points "
            f"(Forearm/Hand/Arm/camera). source_hdf5={hdf5_path}"
        ) from exc


def resolve_background_video(processed_demo_dir: Path, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    sampled_dir = processed_demo_dir / "00_sampled_videos"
    if sampled_dir.exists():
        candidates.extend(sorted(sampled_dir.glob("*.mp4")))
    candidates.extend(
        [
            processed_demo_dir / "inpaint_processor" / "video_human_inpaint_propainter_sam3_qwen_d5.mkv",
            processed_demo_dir / "video_rgb_imgs.mkv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise InputValidationError(f"No background/sample video found near {processed_demo_dir}")


def write_failed_validation(output_dir: Path, error: Exception, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "stage": "qwenrobot_canonical_render_input_validation",
        "valid": False,
        "error_type": type(error).__name__,
        "error": str(error),
        **payload,
    }
    path = output_dir / "FAILED_INPUT_VALIDATION.json"
    write_json(path, failure)
    return path


def resolve_and_validate_inputs(args: argparse.Namespace) -> CanonicalInputs:
    processed_demo_dir = args.processed_demo_dir.resolve()
    trajectory_npz = args.trajectory_npz.resolve()
    output_dir = args.output_dir.resolve()
    if not trajectory_npz.exists():
        raise InputValidationError(f"trajectory_npz does not exist: {trajectory_npz}")
    source_hdf5, source_origin = resolve_source_hdf5(
        explicit_source_hdf5=args.source_hdf5,
        processed_demo_dir=processed_demo_dir,
        trajectory_npz=trajectory_npz,
    )
    validate_hdf5_source(source_hdf5)
    background_video = resolve_background_video(processed_demo_dir, args.background_video)
    return CanonicalInputs(
        processed_demo_dir=processed_demo_dir,
        trajectory_npz=trajectory_npz,
        output_dir=output_dir,
        source_hdf5=source_hdf5,
        source_hdf5_origin=source_origin,
        background_video=background_video,
    )


def load_trajectory_actions(trajectory_npz: Path, max_frames: int | None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    traj = np.load(trajectory_npz, allow_pickle=False)
    required = [
        "left_ee_pos_camera",
        "right_ee_pos_camera",
        "left_ee_rot_camera",
        "right_ee_rot_camera",
        "left_gripper_width",
        "right_gripper_width",
        "frame_indices",
        "camera_intrinsic",
    ]
    missing = [key for key in required if key not in traj.files]
    if missing:
        raise InputValidationError(f"trajectory_npz missing required arrays: {missing}")
    n_frames = len(traj["frame_indices"])
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))
    actions = {
        "left_pos": traj["left_ee_pos_camera"][:n_frames].astype(np.float64),
        "right_pos": traj["right_ee_pos_camera"][:n_frames].astype(np.float64),
        "left_rot": traj["left_ee_rot_camera"][:n_frames].astype(np.float64),
        "right_rot": traj["right_ee_rot_camera"][:n_frames].astype(np.float64),
        "left_width": traj["left_gripper_width"][:n_frames].astype(np.float64),
        "right_width": traj["right_gripper_width"][:n_frames].astype(np.float64),
    }
    frame_indices = traj["frame_indices"][:n_frames].astype(np.int64)
    intrinsic = traj["camera_intrinsic"].astype(np.float64)
    if intrinsic.shape != (3, 3):
        raise InputValidationError(f"camera_intrinsic must be shape (3,3), got {intrinsic.shape}")
    return actions, frame_indices, intrinsic


def load_hdf5_body_points(hdf5_path: Path, frame_indices: np.ndarray, n_frames: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    with h5py.File(str(hdf5_path), "r") as h5:
        camera = h5["transforms/camera"][frame_indices].astype(np.float64)
        for side in ("left", "right"):
            for part in ("Forearm", "Hand", "Arm"):
                key = f"transforms/{side}{part}"
                points_world = h5[key][frame_indices, :3, 3].astype(np.float64)
                out[side][part] = world_points_to_camera(points_world, camera)[:n_frames]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a canonical Kinova3+Robotiq85 robot cache without compositing.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--source-hdf5", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-xml", type=Path, default=None)
    parser.add_argument("--background-video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=68)
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
    parser.add_argument("--determinism-label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        inputs = resolve_and_validate_inputs(args)
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        path = write_failed_validation(
            output_dir,
            exc,
            {
                "processed_demo_dir": str(args.processed_demo_dir),
                "trajectory_npz": str(args.trajectory_npz),
                "source_hdf5_arg": str(args.source_hdf5) if args.source_hdf5 is not None else None,
            },
        )
        print(path)
        raise SystemExit(2) from exc

    from phantom.qwenrobot.mujoco_utils import (
        body_id,
        configure_camera_space_zed,
        export_phantom_kinova_xml,
        load_model,
        render_rgb_mask_depth,
        set_initial_kinova_qpos,
        set_robotiq85_width,
        solve_arm_ik_selected,
        style_kinova_robot,
    )

    xml_path = (args.compiled_xml or inputs.output_dir / "phantom_kinova3_robotiq85_shoulders.xml").resolve()
    export_phantom_kinova_xml(xml_path)

    background = read_video_frames(inputs.background_video)
    if args.max_frames is not None:
        background = background[: args.max_frames]
    actions, frame_indices, intrinsic = load_trajectory_actions(inputs.trajectory_npz, args.max_frames)
    n_frames = min(len(background), len(actions["left_pos"]), len(actions["right_pos"]), len(frame_indices))
    if n_frames <= 0:
        raise InputValidationError("No frames available after loading background and trajectory")
    background = background[:n_frames]
    frame_indices = frame_indices[:n_frames]
    for key in list(actions):
        actions[key] = actions[key][:n_frames]
    height, width = background[0].shape[:2]

    body_points = load_hdf5_body_points(inputs.source_hdf5, frame_indices, n_frames)
    left_root, left_root_pixel = fixed_root_from_forearm_border(
        actions["left_pos"],
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
        actions["right_pos"],
        body_points["right"]["Forearm"],
        body_points["right"]["Arm"],
        intrinsic,
        width,
        height,
        "right",
        border_pad_px=args.root_border_pad_px,
        depth_offset_m=args.root_depth_offset_m,
    )

    mujoco, model, data = load_model(xml_path)
    style_kinova_robot(mujoco, model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera_name = configure_camera_space_zed(mujoco, model, fovy_from_intrinsic(intrinsic, height))
    model.body_pos[body_id(mujoco, model, "robot0_base")] = right_root
    model.body_quat[body_id(mujoco, model, "robot0_base")] = base_quat(
        right_root, np.median(actions["right_pos"], axis=0), "right", args.base_axis
    )
    model.body_pos[body_id(mujoco, model, "robot1_base")] = left_root
    model.body_quat[body_id(mujoco, model, "robot1_base")] = base_quat(
        left_root, np.median(actions["left_pos"], axis=0), "left", args.base_axis
    )

    right_ee_id = body_id(mujoco, model, "gripper0_eef")
    left_ee_id = body_id(mujoco, model, "gripper1_eef")
    link_ids = {
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
    set_initial_kinova_qpos(mujoco, model, data)
    previous_qpos = data.qpos.copy()
    robot_rgb: list[np.ndarray] = []
    robot_mask: list[np.ndarray] = []
    robot_depth: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    link_weights = (args.half_arm_weight, args.forearm_weight, args.wrist_weight)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

    for frame_i in tqdm(range(n_frames), desc="Canonical robot render"):
        set_robotiq85_width(mujoco, model, data, "gripper0", float(actions["right_width"][frame_i]))
        set_robotiq85_width(mujoco, model, data, "gripper1", float(actions["left_width"][frame_i]))
        right = solve_arm_ik_selected(
            mujoco,
            model,
            data,
            target_pos=actions["right_pos"][frame_i],
            target_rot=actions["right_rot"][frame_i],
            ee_body_id=right_ee_id,
            link_targets=human_link_targets(
                body_points,
                "right",
                frame_i,
                root=right_root,
                half_arm_body_id=link_ids["right"][0],
                forearm_body_id=link_ids["right"][1],
                wrist_body_id=link_ids["right"][2],
                weights=link_weights,
            ),
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
            link_targets=human_link_targets(
                body_points,
                "left",
                frame_i,
                root=left_root,
                half_arm_body_id=link_ids["left"][0],
                forearm_body_id=link_ids["left"][1],
                wrist_body_id=link_ids["left"][2],
                weights=link_weights,
            ),
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
        rgb, mask, depth = render_rgb_mask_depth(mujoco, renderer, model, data, camera_name)
        qpos_delta_norm = float(np.linalg.norm(data.qpos - qpos_rows[-1])) if qpos_rows else 0.0
        qpos_rows.append(data.qpos.copy())
        robot_rgb.append(rgb)
        robot_mask.append(mask)
        robot_depth.append(depth)
        metrics.append(
            {
                "frame": int(frame_i),
                "source_frame": int(frame_indices[frame_i]),
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
                "qpos_delta_norm": qpos_delta_norm,
                "robot_mask_area": float(mask.mean()),
            }
        )
    renderer.close()

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    robot_rgb_arr = np.asarray(robot_rgb, dtype=np.uint8)
    robot_mask_arr = np.asarray(robot_mask, dtype=bool)
    robot_depth_arr = np.asarray(robot_depth, dtype=np.float32)
    qpos_arr = np.asarray(qpos_rows, dtype=np.float32)
    key_metrics = {str(row["frame"]): row for row in metrics if row["frame"] in KEY_FRAMES}
    cache_hash = array_hash(robot_rgb_arr, robot_mask_arr, robot_depth_arr, qpos_arr)
    canonical_npz = inputs.output_dir / "robot_render_canonical.npz"
    np.savez_compressed(
        canonical_npz,
        robot_rgb=robot_rgb_arr,
        robot_mask=robot_mask_arr,
        robot_depth=robot_depth_arr,
        qpos=qpos_arr,
        frame_indices=frame_indices.astype(np.int64),
        left_target_width=actions["left_width"].astype(np.float32),
        right_target_width=actions["right_width"].astype(np.float32),
        key_frame_metrics_json=json.dumps(key_metrics),
        frame_metrics_json=json.dumps(metrics),
        cache_hash=cache_hash,
    )
    robot_video = inputs.output_dir / "video_robot_Kinova3_shoulders_canonical.mp4"
    save_video(robot_rgb, robot_video, args.fps)
    summary = {
        "stage": "qwenrobot_canonical_render",
        "valid": True,
        "processed_demo_dir": str(inputs.processed_demo_dir),
        "trajectory_npz": str(inputs.trajectory_npz),
        "source_hdf5": str(inputs.source_hdf5),
        "source_hdf5_origin": inputs.source_hdf5_origin,
        "background_video": str(inputs.background_video),
        "depth_input": None,
        "output_dir": str(inputs.output_dir),
        "compiled_xml": str(xml_path),
        "frames": int(n_frames),
        "fps": float(args.fps),
        "resolution": [int(width), int(height)],
        "frame_indices": frame_indices.astype(int).tolist(),
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
        "key_frame_metrics": key_metrics,
        "canonical_npz": str(canonical_npz),
        "robot_video": str(robot_video),
        "cache_hash": cache_hash,
        "determinism_label": args.determinism_label,
        "visual_alignment_note": "SAM/ProPainter inputs, when used downstream, are SAM3-family approximate implementation artifacts.",
    }
    summary_path = inputs.output_dir / "canonical_render_summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
