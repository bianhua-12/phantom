from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy as media
import numpy as np

from phantom.qwenrobot.prepare_egodex_for_phantom import REPO_ROOT
from phantom.utils.image_utils import get_intrinsics_from_json, get_transformation_matrix_from_extrinsics


HAND_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

DEFAULT_FRAMES = (0, 120, 240, 360, 520, 680, 840, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether EgoDex hand keypoints, Phantom action targets, "
            "and rendered robot overlays are in the same image frame."
        )
    )
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=list(DEFAULT_FRAMES))
    parser.add_argument("--output-prefix", type=str, default="phantom_alignment_diagnostic")
    parser.add_argument("--camera-intrinsics", type=Path, default=None)
    parser.add_argument(
        "--camera-extrinsics",
        type=Path,
        default=REPO_ROOT / "phantom" / "camera" / "camera_extrinsics_ego_bimanual_shoulders.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_video(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return media.read_video(path).astype(np.uint8)


def find_intrinsics(processed_demo_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [processed_demo_dir / "egodex_camera_intrinsics.json"]
    if len(processed_demo_dir.parents) >= 3:
        candidates.append(processed_demo_dir.parents[2] / "egodex_camera_intrinsics.json")
    manifest_path = processed_demo_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if isinstance(manifest, dict) and "raw_demo_dir" in manifest:
            candidates.append(Path(str(manifest["raw_demo_dir"])) / "egodex_camera_intrinsics.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find egodex_camera_intrinsics.json near {processed_demo_dir}")


def project_robot_points(points_robot: np.ndarray, intrinsics: np.ndarray, t_cam_to_robot: np.ndarray) -> np.ndarray:
    robot_to_cam = np.linalg.inv(t_cam_to_robot)
    pts_h = np.concatenate([points_robot, np.ones((len(points_robot), 1), dtype=points_robot.dtype)], axis=1)
    pts_cam = (robot_to_cam @ pts_h.T).T[:, :3]
    z = np.maximum(pts_cam[:, 2], 1e-6)
    u = intrinsics[0, 0] * pts_cam[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * pts_cam[:, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def hand_proxy_2d(kpts_2d: np.ndarray) -> np.ndarray:
    tips = kpts_2d[:, [4, 8, 12]]
    return np.nanmean(tips, axis=1).astype(np.float32)


def draw_hand(frame: np.ndarray, kpts: np.ndarray, color: tuple[int, int, int]) -> None:
    for chain in HAND_CHAINS:
        pts = kpts[chain]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.isfinite(a).all() and np.isfinite(b).all():
                cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 2, cv2.LINE_AA)
        for p in pts:
            if np.isfinite(p).all():
                cv2.circle(frame, tuple(np.round(p).astype(int)), 2, color, -1, cv2.LINE_AA)


def draw_point(frame: np.ndarray, point: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    if not np.isfinite(point).all():
        return
    p = tuple(np.round(point).astype(int))
    cv2.circle(frame, p, 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 220), 26), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def load_actions(processed_demo_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    action_path = processed_demo_dir / "action_processor" / f"actions_{side}_shoulders.npz"
    smooth_path = processed_demo_dir / "smoothing_processor" / f"smoothed_actions_{side}_shoulders.npz"
    action = np.load(action_path, allow_pickle=True)
    smooth = np.load(smooth_path)
    return action["union_indices"].astype(int), smooth["ee_pts"].astype(np.float64)


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    intrinsics_path = find_intrinsics(processed_demo_dir, args.camera_intrinsics)
    intrinsics, _ = get_intrinsics_from_json(str(intrinsics_path))
    extrinsics = load_json(args.camera_extrinsics)
    t_cam_to_robot = get_transformation_matrix_from_extrinsics(extrinsics)

    left_hand = np.load(processed_demo_dir / "hand_processor" / "hand_data_left.npz")
    right_hand = np.load(processed_demo_dir / "hand_processor" / "hand_data_right.npz")
    union_left, ee_left = load_actions(processed_demo_dir, "left")
    union_right, ee_right = load_actions(processed_demo_dir, "right")
    if not np.array_equal(union_left, union_right):
        raise ValueError("Left and right action union indices differ")

    ee_left_2d = project_robot_points(ee_left, intrinsics, t_cam_to_robot)
    ee_right_2d = project_robot_points(ee_right, intrinsics, t_cam_to_robot)
    left_proxy = hand_proxy_2d(left_hand["kpts_2d"])
    right_proxy = hand_proxy_2d(right_hand["kpts_2d"])
    left_proxy_on_union = left_proxy[union_left]
    right_proxy_on_union = right_proxy[union_left]

    left_err = np.linalg.norm(ee_left_2d - left_proxy_on_union, axis=1)
    right_err = np.linalg.norm(ee_right_2d - right_proxy_on_union, axis=1)
    left_detected = left_hand["hand_detected"][union_left].astype(bool)
    right_detected = right_hand["hand_detected"][union_left].astype(bool)

    raw_video = load_video(processed_demo_dir / "video_L.mp4")
    overlay_path = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4"
    if not overlay_path.exists():
        overlay_path = processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4"
    overlay_video = load_video(overlay_path) if overlay_path.exists() else None

    frames = [idx for idx in args.frames if 0 <= idx < len(union_left) and idx < len(raw_video)]
    if not frames:
        raise ValueError("No requested frames are in range")

    rows: list[np.ndarray] = []
    raw_cells: list[np.ndarray] = []
    overlay_cells: list[np.ndarray] = []
    for idx in frames:
        source_idx = union_left[idx]
        raw = raw_video[source_idx].copy()
        draw_hand(raw, left_hand["kpts_2d"][source_idx], (80, 220, 80))
        draw_hand(raw, right_hand["kpts_2d"][source_idx], (80, 140, 255))
        draw_point(raw, ee_left_2d[idx], (0, 255, 255), "L ee")
        draw_point(raw, ee_right_2d[idx], (255, 255, 0), "R ee")
        raw_cells.append(label(raw, f"raw/keypoints f{source_idx}"))

        if overlay_video is not None and idx < len(overlay_video):
            over = overlay_video[idx].copy()
            draw_point(over, ee_left_2d[idx], (0, 255, 255), "L ee")
            draw_point(over, ee_right_2d[idx], (255, 255, 0), "R ee")
            overlay_cells.append(label(over, f"robot overlay f{source_idx}"))

    rows.append(np.concatenate(raw_cells, axis=1))
    if overlay_cells:
        rows.append(np.concatenate(overlay_cells, axis=1))

    montage = np.concatenate(rows, axis=0)
    out_image = processed_demo_dir / f"{args.output_prefix}.jpg"
    cv2.imwrite(str(out_image), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "camera_extrinsics": str(args.camera_extrinsics),
        "frames": frames,
        "union_frame_count": int(len(union_left)),
        "left_detected_count": int(left_detected.sum()),
        "right_detected_count": int(right_detected.sum()),
        "left_ee_to_tip_proxy_px_mean_detected": float(np.nanmean(left_err[left_detected])),
        "left_ee_to_tip_proxy_px_median_detected": float(np.nanmedian(left_err[left_detected])),
        "right_ee_to_tip_proxy_px_mean_detected": float(np.nanmean(right_err[right_detected])),
        "right_ee_to_tip_proxy_px_median_detected": float(np.nanmedian(right_err[right_detected])),
        "image": str(out_image),
        "overlay_video": str(overlay_path) if overlay_path.exists() else None,
    }
    out_json = processed_demo_dir / f"{args.output_prefix}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
