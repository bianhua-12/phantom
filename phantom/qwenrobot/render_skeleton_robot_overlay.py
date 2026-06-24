from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
from scipy.signal import savgol_filter


DEFAULT_HDF5 = Path(
    "/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_demo10_raw/extra/assemble_disassemble_jigsaw_puzzle/0.hdf5"
)
DEFAULT_VIDEO = DEFAULT_HDF5.with_suffix(".mp4")
DEFAULT_OUTPUT_DIR = Path("/mnt/project_rlinf/jlchen/code/phantom_reference/outputs/qwenrobot_skeleton_overlay")


def make_frame_indices(total: int, stride: int, max_frames: int | None) -> np.ndarray:
    indices = np.arange(0, total, max(1, int(stride)), dtype=np.int64)
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    return indices


def world_to_camera(points_world: np.ndarray, camera_transforms: np.ndarray) -> np.ndarray:
    camera_pos = camera_transforms[:, :3, 3]
    camera_rot = camera_transforms[:, :3, :3]
    return np.einsum("nji,nkj->nki", camera_rot, points_world - camera_pos[:, None, :]).astype(np.float32)


def project(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = points_camera[..., 2]
    safe_z = np.maximum(z, 1e-6)
    u = intrinsic[0, 0] * points_camera[..., 0] / safe_z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_camera[..., 1] / safe_z + intrinsic[1, 2]
    return np.stack([u, v, z], axis=-1).astype(np.float32)


def read_projected_tracks(h5_path: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    keys = []
    for side in ("left", "right"):
        keys.extend(
            [
                f"{side}Forearm",
                f"{side}Hand",
                f"{side}ThumbTip",
                f"{side}IndexFingerTip",
                f"{side}MiddleFingerTip",
            ]
        )
    with h5py.File(h5_path, "r") as h5:
        intrinsic = h5["camera/intrinsic"][()].astype(np.float32)
        camera = h5["transforms/camera"][indices].astype(np.float32)
        tracks = {"camera_intrinsic": intrinsic}
        for key in keys:
            points_world = h5[f"transforms/{key}"][indices, :3, 3].astype(np.float32)[:, None, :]
            tracks[key] = project(world_to_camera(points_world, camera), intrinsic)[:, 0]
    return tracks


def smooth_track(track: np.ndarray, window: int = 17) -> np.ndarray:
    out = track.astype(np.float32, copy=True)
    if len(out) < 7:
        return out
    win = min(window, len(out) if len(out) % 2 else len(out) - 1)
    if win < 5:
        return out
    valid = out[:, 2] > 0.02
    for dim in (0, 1):
        values = out[:, dim].copy()
        if valid.sum() >= win:
            values[valid] = savgol_filter(values[valid], win, 2, mode="interp")
        out[:, dim] = values
    return out


def clip_ray_to_image(start: np.ndarray, through: np.ndarray, width: int, height: int, side: str) -> np.ndarray:
    direction = through - start
    if np.linalg.norm(direction) < 1e-6:
        return np.asarray([0.0, height * 0.85], dtype=np.float32) if side == "left" else np.asarray([width - 1.0, height * 0.85], dtype=np.float32)
    candidates = []
    for x in (0.0, float(width - 1)):
        if abs(direction[0]) > 1e-6:
            t = (x - start[0]) / direction[0]
            y = start[1] + t * direction[1]
            if t > 0 and -height * 0.25 <= y <= height * 1.25:
                candidates.append((t, np.asarray([x, np.clip(y, 0, height - 1)], dtype=np.float32)))
    for y in (0.0, float(height - 1)):
        if abs(direction[1]) > 1e-6:
            t = (y - start[1]) / direction[1]
            x = start[0] + t * direction[0]
            if t > 0 and -width * 0.25 <= x <= width * 1.25:
                candidates.append((t, np.asarray([np.clip(x, 0, width - 1), y], dtype=np.float32)))
    if not candidates:
        return np.asarray([0.0, height * 0.85], dtype=np.float32) if side == "left" else np.asarray([width - 1.0, height * 0.85], dtype=np.float32)
    return max(candidates, key=lambda item: item[0])[1]


def draw_capsule(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    p0i = tuple(np.round(p0).astype(int))
    p1i = tuple(np.round(p1).astype(int))
    cv2.line(img, p0i, p1i, (24, 28, 31), radius * 2 + 14, cv2.LINE_AA)
    cv2.circle(img, p0i, radius + 7, (24, 28, 31), -1, cv2.LINE_AA)
    cv2.circle(img, p1i, radius + 7, (24, 28, 31), -1, cv2.LINE_AA)
    cv2.line(img, p0i, p1i, color, radius * 2, cv2.LINE_AA)
    cv2.circle(img, p0i, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, p1i, radius, color, -1, cv2.LINE_AA)
    highlight = tuple(min(255, int(c + 45)) for c in color)
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / norm
        offset = normal * max(2, radius // 3)
        cv2.line(img, tuple(np.round(p0 + offset).astype(int)), tuple(np.round(p1 + offset).astype(int)), highlight, max(2, radius // 4), cv2.LINE_AA)


def draw_joint(img: np.ndarray, point: np.ndarray, radius: int) -> None:
    center = tuple(np.round(point).astype(int))
    cv2.circle(img, center, radius + 9, (16, 18, 20), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius + 4, (65, 70, 74), -1, cv2.LINE_AA)
    cv2.circle(img, center, max(3, radius // 2), (150, 155, 160), -1, cv2.LINE_AA)


def draw_gripper(img: np.ndarray, center: np.ndarray, thumb: np.ndarray, virtual_finger: np.ndarray, scale: float) -> None:
    jaw_axis = thumb - virtual_finger
    norm = np.linalg.norm(jaw_axis)
    if norm < 1e-6:
        jaw_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        norm = 1.0
    jaw_axis = jaw_axis / norm
    approach = np.asarray([-jaw_axis[1], jaw_axis[0]], dtype=np.float32)
    palm = center - approach * (42.0 * scale)
    jaw_len = 68.0 * scale
    jaw_gap = float(np.clip(norm * 0.55, 30.0 * scale, 105.0 * scale))
    cv2.line(img, tuple(np.round(center).astype(int)), tuple(np.round(palm).astype(int)), (18, 20, 23), int(34 * scale), cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(center).astype(int)), int(34 * scale), (22, 24, 27), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(center).astype(int)), int(23 * scale), (78, 82, 86), -1, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        base = center + sign * jaw_axis * (jaw_gap * 0.5)
        tip = base + approach * jaw_len
        cv2.line(img, tuple(np.round(base).astype(int)), tuple(np.round(tip).astype(int)), (10, 12, 14), int(18 * scale), cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(base).astype(int)), int(12 * scale), (45, 48, 52), -1, cv2.LINE_AA)


def draw_side_robot(
    img: np.ndarray,
    side: str,
    forearm: np.ndarray,
    hand: np.ndarray,
    thumb: np.ndarray,
    index: np.ndarray,
    middle: np.ndarray,
) -> bool:
    height, width = img.shape[:2]
    geometry = compute_side_geometry(side, forearm, hand, thumb, index, middle, width, height)
    if geometry is None:
        return False
    draw_side_robot_geometry(img, geometry)
    return True


def compute_side_geometry(
    side: str,
    forearm: np.ndarray,
    hand: np.ndarray,
    thumb: np.ndarray,
    index: np.ndarray,
    middle: np.ndarray,
    width: int,
    height: int,
) -> dict[str, np.ndarray | float | int] | None:
    if hand[2] <= 0.02 or thumb[2] <= 0.02 or index[2] <= 0.02 or middle[2] <= 0.02:
        return None
    virtual_finger = 0.7 * index[:2] + 0.3 * middle[:2]
    ee = 0.5 * (thumb[:2] + virtual_finger)
    forearm_xy = forearm[:2] if forearm[2] > 0.02 and np.all(np.isfinite(forearm[:2])) else hand[:2] + (hand[:2] - ee)
    base = clip_ray_to_image(ee, forearm_xy, width, height, side)
    direction = ee - base
    length = max(float(np.linalg.norm(direction)), 1.0)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / length
    bend_sign = 1.0 if side == "left" else -1.0
    elbow = base * 0.46 + ee * 0.54 + normal * bend_sign * min(45.0, 0.025 * length)
    scale = height / 1080.0
    link_radius = max(26, int(round(68 * scale)))
    forearm_radius = max(24, int(round(60 * scale)))
    return {
        "base": base,
        "elbow": elbow,
        "ee": ee,
        "thumb": thumb[:2],
        "virtual_finger": virtual_finger,
        "scale": float(scale),
        "link_radius": int(link_radius),
        "forearm_radius": int(forearm_radius),
    }


def draw_side_erase_mask(mask: np.ndarray, geometry: dict[str, np.ndarray | float | int]) -> None:
    base = np.asarray(geometry["base"], dtype=np.float32)
    elbow = np.asarray(geometry["elbow"], dtype=np.float32)
    ee = np.asarray(geometry["ee"], dtype=np.float32)
    thumb = np.asarray(geometry["thumb"], dtype=np.float32)
    virtual_finger = np.asarray(geometry["virtual_finger"], dtype=np.float32)
    radius = int(max(int(geometry["link_radius"]), int(geometry["forearm_radius"])) + 28)
    cv2.line(mask, tuple(np.round(base).astype(int)), tuple(np.round(elbow).astype(int)), 255, radius * 2, cv2.LINE_AA)
    cv2.line(mask, tuple(np.round(elbow).astype(int)), tuple(np.round(ee).astype(int)), 255, radius * 2, cv2.LINE_AA)
    cv2.circle(mask, tuple(np.round(base).astype(int)), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, tuple(np.round(elbow).astype(int)), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, tuple(np.round(ee).astype(int)), radius, 255, -1, cv2.LINE_AA)
    cv2.line(mask, tuple(np.round(thumb).astype(int)), tuple(np.round(virtual_finger).astype(int)), 255, max(24, radius), cv2.LINE_AA)


def draw_side_robot_geometry(img: np.ndarray, geometry: dict[str, np.ndarray | float | int]) -> None:
    base = np.asarray(geometry["base"], dtype=np.float32)
    elbow = np.asarray(geometry["elbow"], dtype=np.float32)
    ee = np.asarray(geometry["ee"], dtype=np.float32)
    thumb = np.asarray(geometry["thumb"], dtype=np.float32)
    virtual_finger = np.asarray(geometry["virtual_finger"], dtype=np.float32)
    scale = float(geometry["scale"])
    link_radius = int(geometry["link_radius"])
    forearm_radius = int(geometry["forearm_radius"])
    draw_capsule(img, base, elbow, link_radius, (112, 120, 126))
    draw_capsule(img, elbow, ee, forearm_radius, (132, 140, 146))
    draw_joint(img, base, max(22, int(round(40 * scale))))
    draw_joint(img, elbow, max(20, int(round(34 * scale))))
    draw_joint(img, ee, max(18, int(round(30 * scale))))
    draw_gripper(img, ee, thumb, virtual_finger, scale)


def erase_human_arm(frame: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none" or not np.any(mask):
        return frame.copy()
    if mode == "inpaint":
        return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(frame, (91, 91), 0)
    out = frame.copy()
    soft = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    out[:] = (out.astype(np.float32) * (1.0 - soft[..., None]) + blurred.astype(np.float32) * soft[..., None]).astype(np.uint8)
    return out


def put_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
    return out


def write_montage(raw_samples: list[np.ndarray], overlay_samples: list[np.ndarray], frame_ids: list[int], output_path: Path) -> None:
    raw_row = np.concatenate([put_label(frame, f"raw {idx}") for frame, idx in zip(raw_samples, frame_ids)], axis=1)
    overlay_row = np.concatenate([put_label(frame, f"robot {idx}") for frame, idx in zip(overlay_samples, frame_ids)], axis=1)
    montage = np.concatenate([raw_row, overlay_row], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, montage, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a skeleton-aligned bimanual mechanical overlay for EgoDex.")
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--erase-mode", choices=("blur", "inpaint", "none"), default="blur")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = make_frame_indices(total, args.frame_stride, args.max_frames)
    tracks = read_projected_tracks(args.hdf5, indices)
    for key, value in list(tracks.items()):
        if key != "camera_intrinsic":
            tracks[key] = smooth_track(value)

    output_video = args.output_dir / "skeleton_robot_overlay.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_video}")

    sample_positions = np.linspace(0, len(indices) - 1, min(6, len(indices)), dtype=int).tolist()
    raw_samples: list[np.ndarray] = []
    overlay_samples: list[np.ndarray] = []
    valid_counts = {"left": 0, "right": 0}
    try:
        for out_i, src_i in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_i))
            ok, frame_bgr = cap.read()
            if not ok:
                break
            geometries: list[tuple[str, dict[str, np.ndarray | float | int]]] = []
            erase_mask = np.zeros((height, width), dtype=np.uint8)
            for side in ("left", "right"):
                geometry = compute_side_geometry(
                    side,
                    tracks[f"{side}Forearm"][out_i],
                    tracks[f"{side}Hand"][out_i],
                    tracks[f"{side}ThumbTip"][out_i],
                    tracks[f"{side}IndexFingerTip"][out_i],
                    tracks[f"{side}MiddleFingerTip"][out_i],
                    width,
                    height,
                )
                if geometry is not None:
                    geometries.append((side, geometry))
                    draw_side_erase_mask(erase_mask, geometry)
                    valid_counts[side] += 1
            overlay = erase_human_arm(frame_bgr, erase_mask, args.erase_mode)
            for _, geometry in geometries:
                draw_side_robot_geometry(overlay, geometry)
            writer.write(overlay)
            if out_i in sample_positions:
                raw_samples.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                overlay_samples.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        writer.release()

    frame_ids = [int(indices[pos]) for pos in sample_positions[: len(raw_samples)]]
    montage_path = args.output_dir / "skeleton_robot_overlay_montage.jpg"
    write_montage(raw_samples, overlay_samples, frame_ids, montage_path)
    summary = {
        "hdf5": str(args.hdf5),
        "video": str(args.video),
        "output_video": str(output_video),
        "montage": str(montage_path),
        "frame_stride": int(args.frame_stride),
        "fps": float(args.fps),
        "selected_frames": int(len(indices)),
        "duration_seconds": float(len(indices) / args.fps),
        "erase_mode": args.erase_mode,
        "source_width": int(width),
        "source_height": int(height),
        "valid_left_frames": int(valid_counts["left"]),
        "valid_right_frames": int(valid_counts["right"]),
    }
    summary_path = args.output_dir / "skeleton_robot_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
