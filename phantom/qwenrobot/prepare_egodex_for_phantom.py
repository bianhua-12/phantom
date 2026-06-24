from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
PHANTOM_ROBOSUITE = REPO_ROOT / "submodules" / "phantom-robosuite"
PHANTOM_ROBOMIMIC = REPO_ROOT / "submodules" / "phantom-robomimic"
DEFAULT_EGODEX_ROOT = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_demo10_raw")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "phantom_egodex_exact_epic"
DEFAULT_DEPTHANYTHING3_ROOT = Path(
    "/mnt/project_rlinf/shchen/code/depth-anything-3"
)
DEFAULT_DEPTHANYTHING3_HF_HOME = Path("/mnt/project_rlinf/shchen/.cache/huggingface")
EGODEX_CAMERA_TO_PHANTOM = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
PHANTOM_EPIC_BASE_T_1 = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.5, 0.0, 0.866, 0.2],
        [-0.866, 0.0, 0.5, 1.50],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
PHANTOM_SHOULDERS_BASE_ENV = {
    # Fixed Kinova shoulder bases from phantom-robosuite's original
    # PhantomBimanual shoulders setup, before BASE_T_1 inverse.
    "right": np.asarray([-0.00656507, -0.14111039, 1.58980033], dtype=np.float64),
    "left": np.asarray([0.0, 0.2, 1.5], dtype=np.float64),
}
QWEN_TO_PHANTOM_TOOL = (
    Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    @ Rotation.from_euler("z", -135.0, degrees=True).as_matrix()
)

HAND_LANDMARK_SUFFIXES = [
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
]

HAND_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

ARM_CHAINS = {
    "left": ["leftShoulder", "leftArm", "leftForearm", "leftHand"],
    "right": ["rightShoulder", "rightArm", "rightForearm", "rightHand"],
}


def add_phantom_submodules_to_path() -> None:
    for path in (PHANTOM_ROBOSUITE, PHANTOM_ROBOMIMIC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_extrinsics_matrix(path: Path) -> np.ndarray:
    camera_extrinsics = json.loads(path.read_text(encoding="utf-8"))
    cam_base_pos = np.asarray(camera_extrinsics[0]["camera_base_pos"], dtype=np.float64)
    cam_base_ori = np.asarray(camera_extrinsics[0]["camera_base_ori"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cam_base_ori.reshape(3, 3)
    transform[:3, 3] = cam_base_pos
    return transform


def transform_keypoints(kpts_3d: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([kpts_3d, np.ones((*kpts_3d.shape[:2], 1), dtype=kpts_3d.dtype)], axis=-1)
    return np.einsum("ij,npj->npi", transform, pts_h)[..., :3]


def normalize_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = np.divide(vectors, np.maximum(norms, 1e-8))
    bad = norms[..., 0] < 1e-8
    if np.any(bad):
        out[bad] = fallback
    return out


def qwen_retarget_hand(kpts_3d_rf: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thumb = kpts_3d_rf[:, 4]
    index = kpts_3d_rf[:, 8]
    middle = kpts_3d_rf[:, 12]
    wrist = kpts_3d_rf[:, 0]

    virtual_finger = 0.7 * index + 0.3 * middle
    widths = np.linalg.norm(thumb - virtual_finger, axis=1)
    positions = 0.5 * (thumb + virtual_finger)

    sign = 1.0 if side == "right" else -1.0
    z_axis = normalize_vectors(sign * (thumb - virtual_finger), np.asarray([0.0, 0.0, 1.0]))
    wrist_to_finger = virtual_finger - wrist
    y_axis = normalize_vectors(np.cross(z_axis, wrist_to_finger), np.asarray([0.0, 1.0, 0.0]))
    x_axis = normalize_vectors(np.cross(y_axis, z_axis), np.asarray([1.0, 0.0, 0.0]))
    y_axis = normalize_vectors(np.cross(z_axis, x_axis), np.asarray([0.0, 1.0, 0.0]))

    rotations = np.stack([x_axis, y_axis, z_axis], axis=-1)
    det = np.linalg.det(rotations)
    if np.any(det < 0.0):
        y_axis[det < 0.0] *= -1.0
        rotations = np.stack([x_axis, y_axis, z_axis], axis=-1)

    # Qwen defines the grasp axis as local z, while the Phantom parallel-jaw
    # grippers open along local x after the downstream +135deg z tool offset.
    qwen_to_phantom_tool = (
        Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
        @ Rotation.from_euler("z", -135.0, degrees=True).as_matrix()
    )
    rotations = rotations @ qwen_to_phantom_tool
    return positions, rotations, widths


def fill_missing_actions(
    detected: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    widths: np.ndarray,
    union_indices: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.where(detected)[0]
    if len(valid_indices) == 0:
        neutral_pos = np.asarray([0.4, 0.5 if side == "left" else -0.5, 0.3], dtype=np.float64)
        neutral_rot = np.eye(3, dtype=np.float64)
        return (
            np.repeat(neutral_pos[None], len(union_indices), axis=0),
            np.repeat(neutral_rot[None], len(union_indices), axis=0),
            np.full(len(union_indices), 0.085, dtype=np.float64),
        )

    last = int(valid_indices[0])
    out_pos = []
    out_rot = []
    out_width = []
    for idx in union_indices:
        if detected[idx]:
            last = int(idx)
        out_pos.append(positions[last])
        out_rot.append(rotations[last])
        out_width.append(widths[last])
    return np.asarray(out_pos), np.asarray(out_rot), np.asarray(out_width)


def smooth_positions_and_widths(values: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    n = len(values)
    if n < 5:
        return values
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        window = polyorder + 2 + ((polyorder + 2) % 2 == 0)
    if window > n:
        return values
    return savgol_filter(values, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def smooth_rotations(rotations: np.ndarray) -> np.ndarray:
    from phantom.processors.smoothing_processor import SmoothingProcessor

    if len(rotations) < 3:
        return rotations
    kernel_size = min(21, len(rotations) if len(rotations) % 2 == 1 else len(rotations) - 1)
    return SmoothingProcessor.gaussian_slerp_smoothing(rotations, sigma=10.0, kernel_size=kernel_size)


def write_qwen_action_files(processed_demo_dir: Path, extrinsics_path: Path) -> None:
    add_phantom_submodules_to_path()
    hand_dir = processed_demo_dir / "hand_processor"
    left = np.load(hand_dir / "hand_data_left.npz")
    right = np.load(hand_dir / "hand_data_right.npz")
    left_detected = left["hand_detected"].astype(bool)
    right_detected = right["hand_detected"].astype(bool)
    union_indices = np.where(left_detected | right_detected)[0]
    if len(union_indices) == 0:
        raise ValueError(f"No detected hand frames in {processed_demo_dir}")

    transform = load_extrinsics_matrix(extrinsics_path)
    outputs = {}
    for side, data, detected in (("left", left, left_detected), ("right", right, right_detected)):
        kpts_3d_rf = transform_keypoints(data["kpts_3d"].astype(np.float64), transform)
        positions, rotations, widths = qwen_retarget_hand(kpts_3d_rf, side)
        pos, rot, width = fill_missing_actions(detected, positions, rotations, widths, union_indices, side)
        outputs[side] = {
            "ee_pts": pos,
            "ee_oris": rot,
            "ee_widths": width,
            "smoothed_ee_pts": smooth_positions_and_widths(pos),
            "smoothed_ee_oris": smooth_rotations(rot),
            "smoothed_ee_widths": smooth_positions_and_widths(width),
        }

    action_dir = processed_demo_dir / "action_processor"
    smoothing_dir = processed_demo_dir / "smoothing_processor"
    action_dir.mkdir(parents=True, exist_ok=True)
    smoothing_dir.mkdir(parents=True, exist_ok=True)
    for side, data in outputs.items():
        np.savez(
            action_dir / f"actions_{side}_shoulders.npz",
            union_indices=union_indices,
            ee_pts=data["ee_pts"],
            ee_oris=data["ee_oris"],
            ee_widths=data["ee_widths"],
        )
        np.savez(
            smoothing_dir / f"smoothed_actions_{side}_shoulders.npz",
            ee_pts=data["smoothed_ee_pts"],
            ee_oris=data["smoothed_ee_oris"],
            ee_widths=data["smoothed_ee_widths"],
        )


def discover_demos(root: Path) -> list[tuple[Path, Path, str]]:
    demos: list[tuple[Path, Path, str]] = []
    for h5_path in sorted(root.rglob("*.hdf5")):
        video_path = h5_path.with_suffix(".mp4")
        if video_path.exists():
            rel_id = h5_path.relative_to(root).with_suffix("").as_posix()
            demos.append((h5_path, video_path, rel_id))
    if not demos:
        raise FileNotFoundError(f"No paired .hdf5/.mp4 demos found under {root}")
    return demos


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def make_frame_indices(
    total_frames: int,
    stride: int,
    max_frames: int | None,
    start_frame: int = 0,
) -> np.ndarray:
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"Invalid start_frame={start_frame} for total_frames={total_frames}")
    indices = np.arange(start_frame, total_frames, max(1, stride), dtype=np.int64)
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) == 0:
        raise ValueError("No frames selected")
    return indices


def world_to_camera(points_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    camera_pos = camera_transforms[:, :3, 3]
    camera_rot = camera_transforms[:, :3, :3]
    return np.einsum("nji,nkj->nki", camera_rot, points_world - camera_pos[:, None, :]).astype(np.float32)


def read_body_points_camera(h5: h5py.File, key: str, indices: np.ndarray) -> np.ndarray:
    camera_transforms = h5["transforms/camera"][indices].astype(np.float32)
    points_world = h5[f"transforms/{key}"][indices, :3, 3].astype(np.float32)
    return world_to_camera(points_world[:, None, :], camera_transforms)[:, 0]


def project_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = np.maximum(points_camera[..., 2], 1e-6)
    u = fx * points_camera[..., 0] / z + cx
    v = fy * points_camera[..., 1] / z + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def output_size_for_resolution(input_resolution: int, source_width: int, source_height: int) -> tuple[int, int]:
    if input_resolution == 256:
        return 456, 256
    if input_resolution == 1080:
        return 1920, 1080
    width = int(round(source_width * input_resolution / source_height))
    return width, input_resolution


def scale_intrinsic(intrinsic: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = intrinsic.astype(np.float64, copy=True)
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled.astype(np.float32)


def read_hand_sequence(h5: h5py.File, side: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = [f"{side}{suffix}" for suffix in HAND_LANDMARK_SUFFIXES]
    points_world = np.stack([h5[f"transforms/{key}"][indices, :3, 3] for key in keys], axis=1).astype(np.float32)
    camera_transforms = h5["transforms/camera"][indices].astype(np.float32)
    points_camera = world_to_camera(points_world, camera_transforms)
    kpts_2d = project_camera_points(points_camera, h5["camera/intrinsic"][()])

    required = [0, 4, 8, 12]
    visible = points_camera[:, required, 2] > 0.05
    if "confidences" in h5:
        confidence = np.stack([h5[f"confidences/{key}"][indices] for key in keys], axis=1)
        confident = confidence[:, required].min(axis=1) > 0.15
    else:
        confident = np.ones(len(indices), dtype=bool)
    hand_detected = confident & visible.all(axis=1)
    return hand_detected.astype(bool), kpts_2d, points_camera


def read_arm_points_2d(h5: h5py.File, indices: np.ndarray) -> dict[str, np.ndarray]:
    intrinsic = h5["camera/intrinsic"][()]
    camera_transforms = h5["transforms/camera"][indices].astype(np.float32)
    arm_points: dict[str, np.ndarray] = {}
    for side, keys in ARM_CHAINS.items():
        points_world = np.stack([h5[f"transforms/{key}"][indices, :3, 3] for key in keys], axis=1).astype(np.float32)
        points_camera = world_to_camera(points_world, camera_transforms)
        points_2d = project_camera_points(points_camera, intrinsic)
        points_2d[points_camera[..., 2] <= 0.05] = np.nan
        arm_points[side] = points_2d
    return arm_points


def save_hand_data(path: Path, hand_detected: np.ndarray, kpts_2d: np.ndarray, kpts_3d: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hand_detected=hand_detected,
        kpts_2d=kpts_2d.astype(np.float32),
        kpts_3d=kpts_3d.astype(np.float32),
        frame_indices=np.arange(len(hand_detected), dtype=np.int64),
    )


def hand_bboxes_from_keypoints(
    hand_detected: np.ndarray,
    kpts_2d: np.ndarray,
    width: int,
    height: int,
    padding: float = 24.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return bboxes_from_points(hand_detected, kpts_2d, width, height, padding)


def bboxes_from_points(
    detected_mask: np.ndarray,
    points_2d: np.ndarray,
    width: int,
    height: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = np.zeros((len(points_2d), 4), dtype=np.float32)
    centers = np.zeros((len(points_2d), 2), dtype=np.float32)
    detected = detected_mask.astype(bool).copy()
    for frame_idx, points in enumerate(points_2d):
        finite = np.isfinite(points).all(axis=1)
        if not detected[frame_idx] or not finite.any():
            detected[frame_idx] = False
            continue
        visible = points[finite]
        x0, y0 = np.maximum(visible.min(axis=0) - padding, [0.0, 0.0])
        x1, y1 = np.minimum(visible.max(axis=0) + padding, [width - 1.0, height - 1.0])
        if x1 <= x0 or y1 <= y0:
            detected[frame_idx] = False
            continue
        bboxes[frame_idx] = np.asarray([x0, y0, x1, y1], dtype=np.float32)
        centers[frame_idx] = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    return detected, bboxes, centers


def arm_seed_bboxes_from_points(
    hand_detected: np.ndarray,
    hand_kpts_2d: np.ndarray,
    arm_points_2d: np.ndarray,
    width: int,
    height: int,
    padding: float = 32.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_points = np.concatenate([hand_kpts_2d, arm_points_2d], axis=1)
    return bboxes_from_points(hand_detected, seed_points, width, height, padding)


def save_sam2_seed_data(
    path: Path,
    left_detected: np.ndarray,
    left_kpts_2d: np.ndarray,
    right_detected: np.ndarray,
    right_kpts_2d: np.ndarray,
    arm_points_2d: dict[str, np.ndarray],
    width: int,
    height: int,
) -> None:
    left_detected, left_bboxes, left_centers = arm_seed_bboxes_from_points(
        left_detected, left_kpts_2d, arm_points_2d["left"], width, height
    )
    right_detected, right_bboxes, right_centers = arm_seed_bboxes_from_points(
        right_detected, right_kpts_2d, arm_points_2d["right"], width, height
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        left_hand_detected=left_detected,
        right_hand_detected=right_detected,
        left_bboxes=left_bboxes,
        right_bboxes=right_bboxes,
        left_bboxes_ctr=left_centers,
        right_bboxes_ctr=right_centers,
        left_bbox_min_dist_to_edge=bbox_center_min_dist_to_edge(left_bboxes, width, height),
        right_bbox_min_dist_to_edge=bbox_center_min_dist_to_edge(right_bboxes, width, height),
    )


def bbox_center_min_dist_to_edge(bboxes: np.ndarray, width: int, height: int) -> np.ndarray:
    centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) * 0.5, (bboxes[:, 1] + bboxes[:, 3]) * 0.5], axis=1)
    dists = np.minimum.reduce([centers[:, 0], centers[:, 1], width - centers[:, 0], height - centers[:, 1]])
    invalid = (bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1])
    dists[invalid] = 0.0
    return dists.astype(np.float32)


def save_bbox_data(
    path: Path,
    left_detected: np.ndarray,
    left_kpts_2d: np.ndarray,
    right_detected: np.ndarray,
    right_kpts_2d: np.ndarray,
    width: int,
    height: int,
) -> None:
    left_detected, left_bboxes, left_centers = hand_bboxes_from_keypoints(left_detected, left_kpts_2d, width, height)
    right_detected, right_bboxes, right_centers = hand_bboxes_from_keypoints(
        right_detected, right_kpts_2d, width, height
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        left_hand_detected=left_detected,
        right_hand_detected=right_detected,
        left_bboxes=left_bboxes,
        right_bboxes=right_bboxes,
        left_bboxes_ctr=left_centers,
        right_bboxes_ctr=right_centers,
        left_bbox_min_dist_to_edge=bbox_center_min_dist_to_edge(left_bboxes, width, height),
        right_bbox_min_dist_to_edge=bbox_center_min_dist_to_edge(right_bboxes, width, height),
    )


def draw_polyline(mask: np.ndarray, points: np.ndarray, thickness: int) -> None:
    finite = np.isfinite(points).all(axis=1)
    for a, b in zip(points[:-1], points[1:]):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        p0 = tuple(np.round(a).astype(int).tolist())
        p1 = tuple(np.round(b).astype(int).tolist())
        cv2.line(mask, p0, p1, 1, thickness, lineType=cv2.LINE_AA)
    for point, ok in zip(points, finite):
        if ok:
            cv2.circle(mask, tuple(np.round(point).astype(int).tolist()), max(2, thickness // 3), 1, -1)


def make_arm_masks(
    height: int,
    width: int,
    left_kpts_2d: np.ndarray,
    right_kpts_2d: np.ndarray,
    arm_points_2d: dict[str, np.ndarray],
) -> np.ndarray:
    masks = np.zeros((len(left_kpts_2d), height, width), dtype=np.uint8)
    side_points = {"left": left_kpts_2d, "right": right_kpts_2d}
    resolution_scale = height / 256.0
    arm_thickness = max(1, int(round(72 * resolution_scale)))
    hand_thickness = max(1, int(round(22 * resolution_scale)))
    dilation = max(1, int(round(21 * resolution_scale)))
    if dilation % 2 == 0:
        dilation += 1
    for frame_idx in range(len(masks)):
        mask = masks[frame_idx]
        for side in ("left", "right"):
            draw_polyline(mask, arm_points_2d[side][frame_idx], thickness=arm_thickness)
            for chain in HAND_CHAINS:
                draw_polyline(mask, side_points[side][frame_idx, chain], thickness=hand_thickness)
        masks[frame_idx] = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    return masks


def draw_keypoints(frame_bgr: np.ndarray, left_kpts_2d: np.ndarray, right_kpts_2d: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    for kpts, color in ((left_kpts_2d, (80, 220, 80)), (right_kpts_2d, (80, 140, 255))):
        for chain in HAND_CHAINS:
            pts = kpts[chain]
            for a, b in zip(pts[:-1], pts[1:]):
                if np.isfinite(a).all() and np.isfinite(b).all():
                    cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 3, cv2.LINE_AA)
            for p in pts:
                if np.isfinite(p).all():
                    cv2.circle(out, tuple(np.round(p).astype(int)), 4, color, -1, cv2.LINE_AA)
    return out


def open_writer(path: Path, fps: float, width: int, height: int, fourcc: str) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def write_sampled_videos(
    source_video: Path,
    indices: np.ndarray,
    output_dir: Path,
    fps: float,
    target_width: int,
    target_height: int,
    left_kpts_2d: np.ndarray,
    right_kpts_2d: np.ndarray,
) -> None:
    video_left = open_writer(output_dir / "video_L.mp4", fps, target_width, target_height, "mp4v")
    video_right = open_writer(output_dir / "video_R.mp4", fps, target_width, target_height, "mp4v")
    video_rgb = open_writer(output_dir / "video_rgb_imgs.mkv", fps, target_width, target_height, "FFV1")
    debug_video = open_writer(output_dir / "debug_hand_keypoints.mp4", fps, target_width, target_height, "mp4v")

    original_images = output_dir / "original_images"
    original_images.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    try:
        for out_idx, src_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {src_idx} from {source_video}")
            if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
                frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
            video_left.write(frame_bgr)
            video_right.write(frame_bgr)
            video_rgb.write(frame_bgr)
            debug_video.write(draw_keypoints(frame_bgr, left_kpts_2d[out_idx], right_kpts_2d[out_idx]))
            cv2.imwrite(str(original_images / f"{out_idx:05d}.jpg"), frame_bgr)
    finally:
        cap.release()
        video_left.release()
        video_right.release()
        video_rgb.release()
        debug_video.release()


def write_intrinsics_json(path: Path, intrinsic: np.ndarray, width: int, height: int) -> None:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    v_fov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    h_fov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    d_fov = math.degrees(2.0 * math.atan(math.hypot(width, height) / (2.0 * ((fx + fy) / 2.0))))
    entry = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "disto": [0.0] * 12,
        "v_fov": v_fov,
        "h_fov": h_fov,
        "d_fov": d_fov,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"left": entry, "right": entry}, indent=2), encoding="utf-8")


def write_egodex_extrinsics_json(
    path: Path,
    left_kpts_3d: np.ndarray,
    right_kpts_3d: np.ndarray,
    target_center_env: np.ndarray,
) -> None:
    anchor_points = np.concatenate(
        [
            left_kpts_3d[:, [0, 4, 8, 12], :],
            right_kpts_3d[:, [0, 4, 8, 12], :],
        ],
        axis=1,
    ).reshape(-1, 3)
    mapped_env = anchor_points @ EGODEX_CAMERA_TO_PHANTOM.T
    translation_env = target_center_env - mapped_env.mean(axis=0)

    base_rot = PHANTOM_EPIC_BASE_T_1[:3, :3]
    base_pos = PHANTOM_EPIC_BASE_T_1[:3, 3]
    action_frame_rot = base_rot.T @ EGODEX_CAMERA_TO_PHANTOM
    action_frame_pos = base_rot.T @ (translation_env - base_pos)
    entry = {
        "camera_base_ori": action_frame_rot.tolist(),
        "camera_base_pos": action_frame_pos.tolist(),
        "target_center_env": target_center_env.tolist(),
        "note": "Generated for EgoDex epic mode: first align EgoDex camera to MuJoCo env (z->env x, -x->env y, -y->env z), then premultiply by BASE_T_1 inverse because Phantom applies BASE_T_1 internally.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry], indent=2), encoding="utf-8")


def solve_rigid_transform_no_scale(src_points: np.ndarray, dst_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_center = src_points.mean(axis=0)
    dst_center = dst_points.mean(axis=0)
    covariance = (src_points - src_center).T @ (dst_points - dst_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_center - rotation @ src_center
    errors = np.linalg.norm((src_points @ rotation.T + translation) - dst_points, axis=1)
    return rotation, translation, errors


def write_arm_anchor_extrinsics_json(
    path: Path,
    h5: h5py.File,
    indices: np.ndarray,
    fallback_extrinsics_path: Path,
    alpha: float = 1.0,
) -> None:
    """Align EgoDex upper-arm anchors to Phantom's fixed shoulders bases.

    The same rigid camera-to-action transform is still used by Phantom's action
    processor and renderer, so projected hand targets stay on the EgoDex hands.
    The difference from the original EPIC extrinsics is that the fixed robot
    roots are now behind / below the egocentric camera, matching EgoDex arms
    entering from the lower side of the frame.
    """
    fallback = json.loads(fallback_extrinsics_path.read_text(encoding="utf-8"))[0]
    fallback_rotation = np.asarray(fallback["camera_base_ori"], dtype=np.float64)
    fallback_translation = np.asarray(fallback["camera_base_pos"], dtype=np.float64)

    base_to_action = np.linalg.inv(PHANTOM_EPIC_BASE_T_1)
    base_action = {
        side: (base_to_action @ np.append(pos, 1.0))[:3]
        for side, pos in PHANTOM_SHOULDERS_BASE_ENV.items()
    }

    src_points = []
    dst_points = []
    labels = []
    for side in ("left", "right"):
        upper_arm = np.nanmedian(read_body_points_camera(h5, f"{side}Arm", indices), axis=0).astype(np.float64)
        hand = np.nanmedian(read_body_points_camera(h5, f"{side}Hand", indices), axis=0).astype(np.float64)
        src_points.extend([upper_arm, hand])
        dst_points.extend(
            [
                base_action[side],
                fallback_rotation @ hand + fallback_translation,
            ]
        )
        labels.extend([f"{side}Arm_to_fixed_base", f"{side}Hand_to_original_target"])

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    anchor_rotation, anchor_translation, errors = solve_rigid_transform_no_scale(src, dst)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha < 1.0:
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([fallback_rotation, anchor_rotation]))
        rotation = slerp([alpha]).as_matrix()[0]
        translation = (1.0 - alpha) * fallback_translation + alpha * anchor_translation
    else:
        rotation = anchor_rotation
        translation = anchor_translation

    entry = {
        "camera_base_ori": rotation.tolist(),
        "camera_base_pos": translation.tolist(),
        "mode": "egodex_arm_anchor_to_original_phantom_shoulders",
        "arm_anchor_alpha": alpha,
        "anchor_camera_base_ori": anchor_rotation.tolist(),
        "anchor_camera_base_pos": anchor_translation.tolist(),
        "fallback_camera_base_ori": fallback_rotation.tolist(),
        "fallback_camera_base_pos": fallback_translation.tolist(),
        "source_points_camera": src.tolist(),
        "target_points_action": dst.tolist(),
        "point_labels": labels,
        "fit_errors_m": errors.tolist(),
        "note": (
            "Generated for the original Phantom fixed Kinova3 shoulders setup. "
            "Upper-arm anchors are aligned to the fixed robot bases; hand anchors "
            "are regularized toward the original EPIC hand targets."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry], indent=2), encoding="utf-8")


def write_root_translation_extrinsics_json(
    path: Path,
    h5: h5py.File,
    indices: np.ndarray,
    fallback_extrinsics_path: Path,
    anchor_key: str = "Arm",
) -> None:
    """Translate the original EPIC camera pose so fixed roots project onto EgoDex arms.

    This keeps Phantom's original camera rotation, Kinova3 model, shoulders setup,
    action processor, smoother, and robot renderer unchanged. Only the dataset
    calibration translation is adjusted so the fixed robot bases are on the same
    image side as the human upper arms before we ask the original controller to
    track the EgoDex hand trajectory.
    """
    fallback = json.loads(fallback_extrinsics_path.read_text(encoding="utf-8"))[0]
    rotation = np.asarray(fallback["camera_base_ori"], dtype=np.float64)
    fallback_translation = np.asarray(fallback["camera_base_pos"], dtype=np.float64)

    base_to_action = np.linalg.inv(PHANTOM_EPIC_BASE_T_1)
    base_action = {
        side: (base_to_action @ np.append(pos, 1.0))[:3]
        for side, pos in PHANTOM_SHOULDERS_BASE_ENV.items()
    }

    translations = []
    source_points = {}
    target_points = {}
    for side in ("left", "right"):
        source = np.nanmedian(read_body_points_camera(h5, f"{side}{anchor_key}", indices), axis=0).astype(np.float64)
        translations.append(base_action[side] - rotation @ source)
        source_points[side] = source.tolist()
        target_points[side] = base_action[side].tolist()

    translation = np.mean(np.asarray(translations), axis=0)
    residuals = {
        side: float(np.linalg.norm(rotation @ np.asarray(source_points[side]) + translation - base_action[side]))
        for side in ("left", "right")
    }

    entry = {
        "camera_base_ori": rotation.tolist(),
        "camera_base_pos": translation.tolist(),
        "mode": "egodex_root_translation_to_original_phantom_shoulders",
        "anchor_key": anchor_key,
        "fallback_camera_base_ori": rotation.tolist(),
        "fallback_camera_base_pos": fallback_translation.tolist(),
        "source_points_camera": source_points,
        "target_points_action": target_points,
        "root_anchor_residuals_m": residuals,
        "note": (
            "Generated for the original Phantom fixed Kinova3 shoulders setup. "
            "Only camera translation is changed; all original Phantom processors "
            "and the original shoulders robot morphology remain unchanged."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry], indent=2), encoding="utf-8")


def get_extrinsics_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "extrinsics_mode", None)
    if mode is not None:
        return str(mode)
    return "original_epic_shoulders" if args.use_original_epic_extrinsics else "generated_egodex_target_center"


def build_phantom_cfg(args: argparse.Namespace, intrinsics_path: Path, extrinsics_path: Path) -> Any:
    depth_for_overlay = bool(getattr(args, "depth_for_overlay", False) or getattr(args, "enable_depth_overlay", False))
    depth_occlusion_margin = float(getattr(args, "depth_occlusion_margin", 0.03))
    scene_depth_scale = float(getattr(args, "scene_depth_scale", 1.0))
    scene_depth_offset = float(getattr(args, "scene_depth_offset", 0.0))
    return OmegaConf.create(
        {
            "debug": False,
            "verbose": False,
            "skip_existing": False,
            "n_processes": 1,
            "data_root_dir": str(args.output_root / "raw"),
            "processed_data_root_dir": str(args.output_root / "processed"),
            "demo_name": args.demo_name,
            "mode": ["action", "smoothing", "robot_inpaint"],
            "demo_num": args.demo_num,
            "debug_cameras": [],
            "input_resolution": args.input_resolution,
            "output_resolution": args.output_resolution,
            "robot": args.robot,
            "gripper": args.gripper,
            "square": False,
            "epic": True,
            "bimanual_setup": args.bimanual_setup,
            "target_hand": "both",
            "constrained_hand": False,
            "depth_for_overlay": depth_for_overlay,
            "depth_occlusion_margin": depth_occlusion_margin,
            "scene_depth_scale": scene_depth_scale,
            "scene_depth_offset": scene_depth_offset,
            "depth_debug_dirname": "depth_overlay_debug",
            "render": False,
            "camera_intrinsics": str(intrinsics_path),
            "camera_extrinsics": str(extrinsics_path),
        }
    )


def run_original_processors(
    cfg: Any,
    demo_num: str,
    include_robot_inpaint: bool,
    retarget: str,
    run_hand_inpaint: bool,
    reuse_hand_inpaint: Path | None,
) -> None:
    add_phantom_submodules_to_path()
    processed_demo_dir = Path(cfg.processed_data_root_dir) / cfg.demo_name / demo_num
    raw_demo_dir = Path(cfg.data_root_dir) / cfg.demo_name / demo_num

    if not processed_demo_dir.exists():
        shutil.copytree(raw_demo_dir, processed_demo_dir)

    if reuse_hand_inpaint is not None:
        target = processed_demo_dir / "inpaint_processor" / "video_human_inpaint.mkv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reuse_hand_inpaint, target)
    elif run_hand_inpaint:
        from phantom.processors.handinpaint_processor import HandInpaintProcessor

        target = processed_demo_dir / "inpaint_processor" / "video_human_inpaint.mkv"
        target.unlink(missing_ok=True)
        HandInpaintProcessor(cfg).process_one_demo(demo_num)
    else:
        target = processed_demo_dir / "inpaint_processor" / "video_human_inpaint.mkv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(processed_demo_dir / "video_rgb_imgs.mkv", target)

    if retarget == "qwen":
        write_qwen_action_files(processed_demo_dir, Path(cfg.camera_extrinsics))
    else:
        from phantom.processors.action_processor import ActionProcessor
        from phantom.processors.smoothing_processor import SmoothingProcessor

        ActionProcessor(cfg).process_one_demo(demo_num)
        SmoothingProcessor(cfg).process_one_demo(demo_num)

    if include_robot_inpaint:
        from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

        RobotInpaintProcessor(cfg).process_one_demo(demo_num)


def run_raw_hand_overlay_preview(cfg: Any, demo_num: str, overwrite: bool) -> Path:
    """Render the same robot trajectory over the original RGB video for coverage checks."""
    add_phantom_submodules_to_path()

    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

    source_demo_dir = Path(cfg.processed_data_root_dir) / cfg.demo_name / demo_num
    preview_demo_name = f"{cfg.demo_name}_rawhand_overlay"
    preview_raw_dir = Path(cfg.data_root_dir) / preview_demo_name / demo_num
    preview_demo_dir = Path(cfg.processed_data_root_dir) / preview_demo_name / demo_num

    if overwrite:
        for path in (preview_raw_dir, preview_demo_dir):
            if path.exists():
                shutil.rmtree(path)

    preview_raw_dir.mkdir(parents=True, exist_ok=True)
    preview_demo_dir.mkdir(parents=True, exist_ok=True)

    for dirname in ("action_processor", "smoothing_processor", "segmentation_processor"):
        src = source_demo_dir / dirname
        dst = preview_demo_dir / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    for filename in ("video_rgb_imgs.mkv", "depth.npy", "adapter_manifest.json"):
        src = source_demo_dir / filename
        if src.exists():
            shutil.copy2(src, preview_demo_dir / filename)

    preview_inpaint = preview_demo_dir / "inpaint_processor"
    if preview_inpaint.exists():
        shutil.rmtree(preview_inpaint)
    preview_inpaint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview_demo_dir / "video_rgb_imgs.mkv", preview_inpaint / "video_human_inpaint.mkv")

    preview_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    preview_cfg.demo_name = preview_demo_name
    RobotInpaintProcessor(preview_cfg).process_one_demo(demo_num)

    suffix = f"_{cfg.robot}_{cfg.bimanual_setup}"
    preview_overlay = preview_demo_dir / f"video_overlay{suffix}.mkv"
    target_overlay = source_demo_dir / f"video_overlay{suffix}_rawhand.mkv"
    shutil.copy2(preview_overlay, target_overlay)

    preview_training = preview_demo_dir / "inpaint_processor" / f"training_data_{cfg.bimanual_setup}.npz"
    if preview_training.exists():
        shutil.copy2(
            preview_training,
            source_demo_dir / "inpaint_processor" / f"training_data_{cfg.bimanual_setup}_rawhand.npz",
        )
    return target_overlay


def run_raw_hand_depth_overlay(
    cfg: Any,
    demo_num: str,
    overwrite: bool,
    base_search: Path | None = None,
) -> Path:
    """Render raw-hand RGB with depth-aware robot occlusion as an isolated middle state."""
    add_phantom_submodules_to_path()

    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

    source_demo_dir = Path(cfg.processed_data_root_dir) / cfg.demo_name / demo_num
    preview_demo_name = f"{cfg.demo_name}_rawhand_depth_overlay"
    preview_raw_dir = Path(cfg.data_root_dir) / preview_demo_name / demo_num
    preview_demo_dir = Path(cfg.processed_data_root_dir) / preview_demo_name / demo_num

    if overwrite:
        for path in (preview_raw_dir, preview_demo_dir):
            if path.exists():
                shutil.rmtree(path)

    preview_raw_dir.mkdir(parents=True, exist_ok=True)
    preview_demo_dir.mkdir(parents=True, exist_ok=True)

    for dirname in ("action_processor", "smoothing_processor", "segmentation_processor"):
        src = source_demo_dir / dirname
        dst = preview_demo_dir / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    for filename in ("video_rgb_imgs.mkv", "depth.npy", "adapter_manifest.json"):
        src = source_demo_dir / filename
        if src.exists():
            shutil.copy2(src, preview_demo_dir / filename)

    preview_inpaint = preview_demo_dir / "inpaint_processor"
    if preview_inpaint.exists():
        shutil.rmtree(preview_inpaint)
    preview_inpaint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview_demo_dir / "video_rgb_imgs.mkv", preview_inpaint / "video_human_inpaint.mkv")

    preview_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    preview_cfg.demo_name = preview_demo_name
    preview_cfg.depth_for_overlay = True
    preview_cfg.depth_occlusion_margin = float(getattr(cfg, "depth_occlusion_margin", 0.03))
    preview_cfg.scene_depth_scale = float(getattr(cfg, "scene_depth_scale", 1.0))
    preview_cfg.scene_depth_offset = float(getattr(cfg, "scene_depth_offset", 0.0))
    preview_cfg.depth_debug_dirname = "depth_overlay_debug_rawhand"
    if base_search is not None:
        payload = json.loads(base_search.read_text(encoding="utf-8"))
        if payload.get("qwen_tool_axis_fix", False):
            apply_qwen_tool_axis_fix_to_action_files(
                preview_demo_dir,
                cfg.bimanual_setup,
            )
        apply_tool_roll_to_action_files(
            preview_demo_dir,
            cfg.bimanual_setup,
            float(payload.get("tool_roll_deg", 0.0)),
        )
        apply_base_search_to_environment(payload)
    try:
        RobotInpaintProcessor(preview_cfg).process_one_demo(demo_num)
    finally:
        if base_search is not None:
            clear_base_search_environment()

    suffix = f"_{cfg.robot}_{cfg.bimanual_setup}"
    preview_overlay = preview_demo_dir / f"video_overlay{suffix}.mkv"
    target_overlay = source_demo_dir / f"video_overlay{suffix}_rawhand_depth.mkv"
    shutil.copy2(preview_overlay, target_overlay)

    source_inpaint = source_demo_dir / "inpaint_processor"
    source_inpaint.mkdir(parents=True, exist_ok=True)
    preview_debug = preview_demo_dir / "inpaint_processor" / "depth_overlay_debug_rawhand"
    target_debug = source_inpaint / "depth_overlay_debug_rawhand"
    if preview_debug.exists():
        if target_debug.exists():
            shutil.rmtree(target_debug)
        shutil.copytree(preview_debug, target_debug)

    preview_training = preview_demo_dir / "inpaint_processor" / f"training_data_{cfg.bimanual_setup}.npz"
    if preview_training.exists():
        shutil.copy2(
            preview_training,
            source_inpaint / f"training_data_{cfg.bimanual_setup}_rawhand_depth.npz",
        )
    return target_overlay


def apply_base_search_to_environment(payload: dict[str, Any]) -> None:
    os.environ["PHANTOM_BIMANUAL_BASE0_OFFSET"] = ",".join(str(v) for v in payload["base0_offset"])
    os.environ["PHANTOM_BIMANUAL_BASE1_OFFSET"] = ",".join(str(v) for v in payload["base1_offset"])
    if "global_yaw_deg" in payload:
        os.environ["PHANTOM_BIMANUAL_GLOBAL_YAW_DEG"] = str(payload["global_yaw_deg"])


def clear_base_search_environment() -> None:
    os.environ.pop("PHANTOM_BIMANUAL_BASE0_OFFSET", None)
    os.environ.pop("PHANTOM_BIMANUAL_BASE1_OFFSET", None)
    os.environ.pop("PHANTOM_BIMANUAL_GLOBAL_YAW_DEG", None)


def run_depthanything_v2(processed_demo_dir: Path, args: argparse.Namespace) -> None:
    if args.depth_checkpoint is None:
        raise ValueError("--depth-checkpoint is required when --depth-mode depthanything-v2")
    cmd = [
        sys.executable,
        "-m",
        "phantom.qwenrobot.depthanything_scene_depth",
        "--processed-demo-dir",
        str(processed_demo_dir),
        "--depthanything-root",
        str(args.depthanything_root),
        "--checkpoint",
        str(args.depth_checkpoint),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_depthanything_v3(processed_demo_dir: Path, intrinsics_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "phantom.qwenrobot.depthanything3_scene_depth",
        "--processed-demo-dir",
        str(processed_demo_dir),
        "--model-id",
        args.depthanything3_model_id,
        "--process-res",
        str(args.depthanything3_process_res),
        "--process-res-method",
        args.depthanything3_process_res_method,
        "--ref-view-strategy",
        args.depthanything3_ref_view_strategy,
        "--chunk-size",
        str(args.depthanything3_chunk_size),
        "--metric-mode",
        args.depthanything3_metric_mode,
        "--camera-intrinsics",
        str(intrinsics_path),
    ]
    if args.depthanything3_root is not None:
        cmd.extend(["--depthanything3-root", str(args.depthanything3_root)])
    if args.depthanything3_hf_home is not None:
        cmd.extend(["--hf-home", str(args.depthanything3_hf_home)])
    if args.depthanything3_focal_length_px is not None:
        cmd.extend(["--focal-length-px", str(args.depthanything3_focal_length_px)])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_phantom_base_search(processed_demo_dir: Path, extrinsics_path: Path, args: argparse.Namespace) -> Path:
    output = processed_demo_dir / "base_search" / "best_base.json"
    cmd = [
        sys.executable,
        "-m",
        "phantom.qwenrobot.search_phantom_base_offsets",
        "--processed-demo-dir",
        str(processed_demo_dir),
        "--camera-extrinsics",
        str(extrinsics_path),
        "--robot",
        args.robot,
        "--gripper",
        args.gripper,
        "--bimanual-setup",
        args.bimanual_setup,
        "--input-resolution",
        str(args.input_resolution),
        "--output-resolution",
        str(args.output_resolution),
        "--output",
        str(output),
        "--search-strategy",
        "staged",
        "--xy-radius",
        str(args.phantom_base_search_xy_radius),
        "--xy-step",
        str(args.phantom_base_search_xy_step),
        "--z-offsets",
        *(str(v) for v in args.phantom_base_search_z_offsets),
        "--yaw-degs",
        *(str(v) for v in args.phantom_base_search_yaw_degs),
        "--tool-roll-degs",
        *(str(v) for v in args.phantom_base_search_tool_roll_degs),
        "--top-offsets",
        str(args.phantom_base_search_top_offsets),
        "--tracking-threshold",
        str(args.phantom_base_search_tracking_threshold),
        "--visual-aware-ranking",
        "--min-valid-ratio-for-visual-sort",
        str(args.phantom_base_search_min_valid_ratio_for_visual_sort),
        "--visual-area-weight",
        str(args.phantom_base_search_visual_area_weight),
        "--bottom-area-weight",
        str(args.phantom_base_search_bottom_area_weight),
        "--bbox-bottom-weight",
        str(args.phantom_base_search_bbox_bottom_weight),
        "--projection-error-weight",
        str(args.phantom_base_search_projection_error_weight),
        "--crossing-penalty-weight",
        str(args.phantom_base_search_crossing_penalty_weight),
    ]
    if args.phantom_base_search_max_bottom_area_for_visual_sort is not None:
        cmd.extend(
            [
                "--max-bottom-area-for-visual-sort",
                str(args.phantom_base_search_max_bottom_area_for_visual_sort),
            ]
        )
    if args.phantom_base_search_max_bbox_bottom_touch_for_visual_sort is not None:
        cmd.extend(
            [
                "--max-bbox-bottom-touch-for-visual-sort",
                str(args.phantom_base_search_max_bbox_bottom_touch_for_visual_sort),
            ]
        )
    if args.phantom_base_search_max_projection_error_for_visual_sort is not None:
        cmd.extend(
            [
                "--max-projection-error-for-visual-sort",
                str(args.phantom_base_search_max_projection_error_for_visual_sort),
            ]
        )
    if getattr(args, "phantom_base_search_independent_offsets", False):
        cmd.append("--independent-base-offsets")
    if args.phantom_base_search_max_candidates is not None:
        cmd.extend(["--max-candidates", str(args.phantom_base_search_max_candidates)])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return output


def apply_tool_roll_to_action_files(processed_demo_dir: Path, bimanual_setup: str, tool_roll_deg: float) -> None:
    if abs(tool_roll_deg) < 1e-12:
        return
    tool_roll = Rotation.from_euler("x", tool_roll_deg, degrees=True).as_matrix()
    for folder, stem in (
        ("action_processor", "actions"),
        ("smoothing_processor", "smoothed_actions"),
    ):
        for side in ("left", "right"):
            path = processed_demo_dir / folder / f"{stem}_{side}_{bimanual_setup}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            if "ee_oris" not in payload:
                continue
            payload["ee_oris"] = np.asarray(payload["ee_oris"]) @ tool_roll
            np.savez(path, **payload)


def apply_qwen_tool_axis_fix_to_action_files(processed_demo_dir: Path, bimanual_setup: str) -> None:
    for folder, stem in (
        ("action_processor", "actions"),
        ("smoothing_processor", "smoothed_actions"),
    ):
        for side in ("left", "right"):
            path = processed_demo_dir / folder / f"{stem}_{side}_{bimanual_setup}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            if "ee_oris" not in payload:
                continue
            payload["ee_oris"] = np.asarray(payload["ee_oris"]) @ QWEN_TO_PHANTOM_TOOL
            np.savez(path, **payload)


def prepare_one_demo(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    demos = discover_demos(args.egodex_root)
    h5_path, video_path, rel_id = demos[args.demo_index]
    info = video_info(video_path)
    target_width, target_height = output_size_for_resolution(args.input_resolution, info["width"], info["height"])
    scale_x = target_width / info["width"]
    scale_y = target_height / info["height"]
    raw_demo_dir = args.output_root / "raw" / args.demo_name / args.demo_num
    processed_demo_dir = args.output_root / "processed" / args.demo_name / args.demo_num

    if args.overwrite:
        for path in (raw_demo_dir, processed_demo_dir):
            if path.exists():
                shutil.rmtree(path)
    raw_demo_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        total_frames = min(info["frames"], h5["transforms/camera"].shape[0])
        indices = make_frame_indices(total_frames, args.frame_stride, args.max_frames, args.start_frame)
        left_detected, left_kpts_2d, left_kpts_3d = read_hand_sequence(h5, "left", indices)
        right_detected, right_kpts_2d, right_kpts_3d = read_hand_sequence(h5, "right", indices)
        arm_points_2d = read_arm_points_2d(h5, indices)
        left_kpts_2d *= np.asarray([scale_x, scale_y], dtype=np.float32)
        right_kpts_2d *= np.asarray([scale_x, scale_y], dtype=np.float32)
        for side in arm_points_2d:
            arm_points_2d[side] *= np.asarray([scale_x, scale_y], dtype=np.float32)
        masks_arm = make_arm_masks(target_height, target_width, left_kpts_2d, right_kpts_2d, arm_points_2d)
        save_hand_data(raw_demo_dir / "hand_processor" / "hand_data_left.npz", left_detected, left_kpts_2d, left_kpts_3d)
        save_hand_data(raw_demo_dir / "hand_processor" / "hand_data_right.npz", right_detected, right_kpts_2d, right_kpts_3d)
        save_bbox_data(
            raw_demo_dir / "bbox_processor" / "bbox_data.npz",
            left_detected,
            left_kpts_2d,
            right_detected,
            right_kpts_2d,
            target_width,
            target_height,
        )
        save_sam2_seed_data(
            raw_demo_dir / "bbox_processor" / "sam2_seed_data.npz",
            left_detected,
            left_kpts_2d,
            right_detected,
            right_kpts_2d,
            arm_points_2d,
            target_width,
            target_height,
        )
        (raw_demo_dir / "segmentation_processor").mkdir(parents=True, exist_ok=True)
        np.save(raw_demo_dir / "segmentation_processor" / "masks_arm.npy", masks_arm)
        np.save(raw_demo_dir / "depth.npy", np.ones((len(indices), target_height, target_width), dtype=np.float32) * 10.0)
        intrinsics_path = raw_demo_dir / "egodex_camera_intrinsics.json"
        extrinsics_path = raw_demo_dir / "egodex_camera_extrinsics_shoulders.json"
        write_intrinsics_json(intrinsics_path, scale_intrinsic(h5["camera/intrinsic"][()], scale_x, scale_y), target_width, target_height)
        extrinsics_mode = get_extrinsics_mode(args)
        if extrinsics_mode == "original_epic_shoulders":
            extrinsics_path = REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json"
        elif extrinsics_mode == "arm_anchor":
            write_arm_anchor_extrinsics_json(
                extrinsics_path,
                h5,
                indices,
                REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json",
                float(getattr(args, "arm_anchor_alpha", 1.0)),
            )
        elif extrinsics_mode == "root_translation":
            write_root_translation_extrinsics_json(
                extrinsics_path,
                h5,
                indices,
                REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json",
                str(getattr(args, "root_anchor_key", "Arm")),
            )
        else:
            write_egodex_extrinsics_json(
                extrinsics_path,
                left_kpts_3d,
                right_kpts_3d,
                np.asarray(args.target_center_env, dtype=np.float64),
            )

    write_sampled_videos(video_path, indices, raw_demo_dir, args.output_fps, target_width, target_height, left_kpts_2d, right_kpts_2d)
    manifest = {
        "source_hdf5": str(h5_path),
        "source_video": str(video_path),
        "source_rel_id": rel_id,
        "raw_demo_dir": str(raw_demo_dir),
        "processed_demo_dir": str(processed_demo_dir),
        "frame_stride": int(args.frame_stride),
        "start_frame": int(args.start_frame),
        "output_fps": float(args.output_fps),
        "output_width": int(target_width),
        "output_height": int(target_height),
        "selected_frame_count": int(len(indices)),
        "selected_source_indices": indices.tolist(),
        "left_detected_frames": int(left_detected.sum()),
        "right_detected_frames": int(right_detected.sum()),
        "robot": args.robot,
        "gripper": args.gripper,
        "bimanual_setup": args.bimanual_setup,
        "retarget": args.retarget,
        "camera_extrinsics": str(extrinsics_path),
        "extrinsics_mode": get_extrinsics_mode(args),
        "hand_inpaint_mode": (
            "reuse"
            if args.reuse_hand_inpaint is not None
            else "run_e2fgvi"
            if args.run_hand_inpaint
            else "raw_video_placeholder"
        ),
        "reuse_hand_inpaint": str(args.reuse_hand_inpaint) if args.reuse_hand_inpaint is not None else None,
        "pipeline": (
            "EgoDex adapter followed by unmodified Phantom Action/Smoothing/RobotInpaint processors"
            if args.retarget == "phantom"
            else "EgoDex adapter followed by Qwen-style action files and Phantom RobotInpaint"
        ),
    }
    (raw_demo_dir / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw_demo_dir, intrinsics_path, extrinsics_path


def load_prepared_demo_context(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    raw_demo_dir = args.output_root / "raw" / args.demo_name / args.demo_num
    processed_demo_dir = args.output_root / "processed" / args.demo_name / args.demo_num
    if not raw_demo_dir.exists():
        raise FileNotFoundError(f"--reuse-prepared-demo requires existing raw demo dir: {raw_demo_dir}")
    if not processed_demo_dir.exists():
        raise FileNotFoundError(f"--reuse-prepared-demo requires existing processed demo dir: {processed_demo_dir}")

    intrinsics_candidates = [
        raw_demo_dir / "egodex_camera_intrinsics.json",
        processed_demo_dir / "egodex_camera_intrinsics.json",
        args.output_root / "egodex_camera_intrinsics.json",
    ]
    intrinsics_path = next((path for path in intrinsics_candidates if path.exists()), None)
    if intrinsics_path is None:
        raise FileNotFoundError(
            "Missing prepared intrinsics; checked: "
            + ", ".join(str(path) for path in intrinsics_candidates)
        )

    manifest_path = raw_demo_dir / "adapter_manifest.json"
    if not manifest_path.exists():
        manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extrinsics_path = Path(manifest.get("camera_extrinsics", ""))
        if not extrinsics_path.is_absolute():
            extrinsics_path = raw_demo_dir / extrinsics_path
    else:
        extrinsics_path = raw_demo_dir / "egodex_camera_extrinsics_shoulders.json"
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Missing prepared extrinsics: {extrinsics_path}")
    return raw_demo_dir, intrinsics_path, extrinsics_path


def parse_args() -> argparse.Namespace:
    def optional_frame_count(value: str) -> int | None:
        if value.lower() in {"none", "null", "full", "-1", "0"}:
            return None
        parsed = int(value)
        if parsed < 0:
            return None
        return parsed

    parser = argparse.ArgumentParser(description="Prepare an EgoDex demo for the original Phantom robot overlay pipeline.")
    parser.add_argument("--egodex-root", type=Path, default=DEFAULT_EGODEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--demo-name", type=str, default="egodex_phantom")
    parser.add_argument("--demo-num", type=str, default="0")
    parser.add_argument("--frame-stride", type=int, default=2, help="2 converts the 30 FPS EgoDex video to 15 FPS.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source-video frame to sample.")
    parser.add_argument("--max-frames", type=optional_frame_count, default=120, help="Use none/full for a full-length selected trajectory.")
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--depth-mode", choices=("placeholder", "depthanything-v2", "depthanything-v3"), default="placeholder")
    parser.add_argument("--depthanything-root", type=Path, default=Path("/mnt/project_rlinf/jlchen/code/WAFT/thirdparty/DepthAnythingV2"))
    parser.add_argument("--depth-checkpoint", type=Path, default=None)
    parser.add_argument("--depthanything3-root", type=Path, default=DEFAULT_DEPTHANYTHING3_ROOT)
    parser.add_argument("--depthanything3-model-id", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--depthanything3-hf-home", type=Path, default=None)
    parser.add_argument("--depthanything3-process-res", type=int, default=504)
    parser.add_argument("--depthanything3-process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--depthanything3-ref-view-strategy", type=str, default="middle")
    parser.add_argument("--depthanything3-chunk-size", type=int, default=16)
    parser.add_argument("--depthanything3-metric-mode", choices=("hand-calibrated", "direct", "focal-scaled"), default="hand-calibrated")
    parser.add_argument("--depthanything3-focal-length-px", type=float, default=None)
    parser.add_argument("--enable-depth-overlay", action="store_true")
    parser.add_argument("--depth-occlusion-margin", type=float, default=0.03)
    parser.add_argument("--scene-depth-scale", type=float, default=1.0)
    parser.add_argument("--scene-depth-offset", type=float, default=0.0)
    parser.add_argument("--run-base-search", action="store_true")
    parser.add_argument("--phantom-base-search-xy-radius", type=float, default=0.60)
    parser.add_argument("--phantom-base-search-xy-step", type=float, default=0.20)
    parser.add_argument(
        "--phantom-base-search-z-offsets",
        type=float,
        nargs="*",
        default=[0.0, 0.20, 0.40],
    )
    parser.add_argument(
        "--phantom-base-search-yaw-degs",
        type=float,
        nargs="*",
        default=[0.0, 15.0, -15.0, 30.0, -30.0],
    )
    parser.add_argument(
        "--phantom-base-search-tool-roll-degs",
        type=float,
        nargs="*",
        default=[0.0, 45.0, -45.0, 90.0, -90.0],
    )
    parser.add_argument("--phantom-base-search-top-offsets", type=int, default=6)
    parser.add_argument("--phantom-base-search-tracking-threshold", type=float, default=0.05)
    parser.add_argument("--phantom-base-search-min-valid-ratio-for-visual-sort", type=float, default=0.80)
    parser.add_argument("--phantom-base-search-visual-area-weight", type=float, default=0.20)
    parser.add_argument("--phantom-base-search-bottom-area-weight", type=float, default=1.20)
    parser.add_argument("--phantom-base-search-bbox-bottom-weight", type=float, default=1.50)
    parser.add_argument("--phantom-base-search-projection-error-weight", type=float, default=0.003)
    parser.add_argument("--phantom-base-search-crossing-penalty-weight", type=float, default=0.006)
    parser.add_argument("--phantom-base-search-max-bottom-area-for-visual-sort", type=float, default=0.25)
    parser.add_argument("--phantom-base-search-max-bbox-bottom-touch-for-visual-sort", type=float, default=None)
    parser.add_argument("--phantom-base-search-max-projection-error-for-visual-sort", type=float, default=70.0)
    parser.add_argument("--phantom-base-search-independent-offsets", action="store_true")
    parser.add_argument(
        "--phantom-base-search-max-candidates",
        type=int,
        default=None,
        help="Optional smoke-test cap on offset candidates before staged yaw/tool search.",
    )
    parser.add_argument("--robot", type=str, default="Kinova3")
    parser.add_argument("--gripper", type=str, default="Robotiq85")
    parser.add_argument("--bimanual-setup", choices=("shoulders", "shoulders1", "shoulders2"), default="shoulders")
    parser.add_argument("--retarget", choices=("phantom", "qwen"), default="phantom")
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=256)
    parser.add_argument(
        "--target-center-env",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.25),
        metavar=("X", "Y", "Z"),
        help="Target center of EgoDex hand keypoints in Phantom/MuJoCo world coordinates before BASE_T_1 inverse.",
    )
    parser.add_argument(
        "--arm-anchor-alpha",
        type=float,
        default=1.0,
        help="Interpolation from original EPIC extrinsics (0) to EgoDex upper-arm anchored extrinsics (1).",
    )
    extrinsics_group = parser.add_mutually_exclusive_group()
    extrinsics_group.add_argument(
        "--use-original-epic-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="original_epic_shoulders",
        default="original_epic_shoulders",
        help="Use Phantom/Masquerade's fixed shoulders camera extrinsics. This is the default original-repo baseline.",
    )
    extrinsics_group.add_argument(
        "--use-generated-egodex-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="generated_egodex_target_center",
        help="Use the experimental EgoDex target-center extrinsics path.",
    )
    extrinsics_group.add_argument(
        "--use-arm-anchor-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="arm_anchor",
        help="Rigidly align EgoDex upper-arm anchors to Phantom's fixed shoulders bases.",
    )
    extrinsics_group.add_argument(
        "--use-root-translation-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="root_translation",
        help="Keep Phantom's EPIC rotation and translate the fixed shoulders roots toward EgoDex arm anchors.",
    )
    parser.add_argument(
        "--root-anchor-key",
        choices=("Shoulder", "Arm", "Forearm"),
        default="Arm",
        help="EgoDex body anchor used by --use-root-translation-extrinsics.",
    )
    hand_inpaint_group = parser.add_mutually_exclusive_group()
    hand_inpaint_group.add_argument(
        "--run-hand-inpaint",
        action="store_true",
        help="Run Phantom's original E2FGVI hand-inpaint stage before robot overlay.",
    )
    hand_inpaint_group.add_argument(
        "--reuse-hand-inpaint",
        type=Path,
        default=None,
        help="Copy an existing video_human_inpaint.mkv with the same frame sequence before robot overlay.",
    )
    parser.add_argument("--skip-robot-inpaint", action="store_true")
    parser.add_argument(
        "--raw-hand-overlay-preview",
        action="store_true",
        help="Also render the robot over the original RGB video for hand-coverage diagnostics.",
    )
    parser.add_argument(
        "--middle-state-raw-depth-overlay",
        action="store_true",
        help=(
            "Run the MarionLepert/phantom middle state: Qwen retarget, raw-hand RGB "
            "as video_human_inpaint, DepthAnything3 scene depth, and depth-aware raw-hand overlay only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-prepared-demo",
        action="store_true",
        help="Skip EgoDex discovery/frame extraction and reuse an existing output-root raw/processed demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.middle_state_raw_depth_overlay:
        args.retarget = "qwen"
        args.depth_mode = "depthanything-v3"
        args.enable_depth_overlay = True
        args.run_hand_inpaint = False
        args.reuse_hand_inpaint = None
        args.skip_robot_inpaint = False
        args.run_base_search = True
        if args.depthanything3_root is None:
            args.depthanything3_root = DEFAULT_DEPTHANYTHING3_ROOT
        if args.depthanything3_hf_home is None:
            args.depthanything3_hf_home = DEFAULT_DEPTHANYTHING3_HF_HOME
    args.output_root = args.output_root.resolve()
    if args.reuse_prepared_demo:
        raw_demo_dir, intrinsics_path, extrinsics_path = load_prepared_demo_context(args)
    else:
        raw_demo_dir, intrinsics_path, extrinsics_path = prepare_one_demo(args)
    cfg = build_phantom_cfg(args, intrinsics_path, extrinsics_path)
    if not args.prepare_only:
        needs_pre_robot_steps = args.depth_mode in {"depthanything-v2", "depthanything-v3"} or args.run_base_search
        run_original_processors(
            cfg,
            args.demo_num,
            include_robot_inpaint=(not args.skip_robot_inpaint and not needs_pre_robot_steps),
            retarget=args.retarget,
            run_hand_inpaint=args.run_hand_inpaint,
            reuse_hand_inpaint=args.reuse_hand_inpaint,
        )
        processed_demo_dir = args.output_root / "processed" / args.demo_name / args.demo_num
        base_search = None
        if args.depth_mode == "depthanything-v2":
            run_depthanything_v2(processed_demo_dir, args)
        elif args.depth_mode == "depthanything-v3":
            run_depthanything_v3(processed_demo_dir, intrinsics_path, args)
        if args.run_base_search:
            base_search = run_phantom_base_search(processed_demo_dir, extrinsics_path, args)
            print(f"base_search={base_search}")
        if not args.skip_robot_inpaint and needs_pre_robot_steps and not args.middle_state_raw_depth_overlay:
            if base_search is not None:
                payload = json.loads(base_search.read_text(encoding="utf-8"))
                apply_base_search_to_environment(payload)
                apply_tool_roll_to_action_files(
                    processed_demo_dir,
                    args.bimanual_setup,
                    float(payload.get("tool_roll_deg", 0.0)),
                )
            from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

            try:
                RobotInpaintProcessor(cfg).process_one_demo(args.demo_num)
            finally:
                clear_base_search_environment()
        if args.middle_state_raw_depth_overlay and not args.skip_robot_inpaint:
            rawhand_depth_overlay = run_raw_hand_depth_overlay(
                cfg,
                args.demo_num,
                overwrite=args.overwrite,
                base_search=base_search,
            )
            print(f"rawhand_depth_overlay_video={rawhand_depth_overlay}")
        elif args.raw_hand_overlay_preview and not args.skip_robot_inpaint:
            rawhand_overlay = run_raw_hand_overlay_preview(cfg, args.demo_num, overwrite=args.overwrite)
            print(f"rawhand_overlay_video={rawhand_overlay}")
    overlay_path = args.output_root / "processed" / args.demo_name / args.demo_num / f"video_overlay_{args.robot}_{args.bimanual_setup}.mkv"
    print(f"prepared_demo={raw_demo_dir}")
    print(f"overlay_video={overlay_path}")


if __name__ == "__main__":
    main()
