from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import numpy as np

from phantom.qwenrobot.prepare_egodex_for_phantom import read_arm_points_2d, video_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite a visual mechanical arm cover over SAM hand/arm regions."
    )
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--clean-video", type=Path, required=True)
    parser.add_argument("--robot-overlay-video", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--montage", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--link-radius", type=float, default=58.0)
    parser.add_argument("--hand-radius", type=float, default=34.0)
    parser.add_argument("--mask-dilation", type=int, default=27)
    parser.add_argument("--robot-diff-threshold", type=float, default=18.0)
    parser.add_argument("--montage-frames", type=int, nargs="*", default=[0, 15, 30, 45, 59])
    return parser.parse_args()


def read_video_rgb(path: Path) -> np.ndarray:
    frames = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return np.asarray(frames, dtype=np.uint8)


def load_arm_points(processed_demo_dir: Path, width: int, height: int) -> dict[str, np.ndarray]:
    manifest = json.loads((processed_demo_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    indices = np.asarray(manifest["selected_source_indices"], dtype=np.int64)
    source_video = Path(manifest["source_video"])
    source_info = video_info(source_video)
    scale = np.asarray([width / source_info["width"], height / source_info["height"]], dtype=np.float32)
    with h5py.File(manifest["source_hdf5"], "r") as h5:
        arm_points = read_arm_points_2d(h5, indices)
    for side in arm_points:
        arm_points[side] = arm_points[side] * scale
    return arm_points


def load_hand_points(processed_demo_dir: Path) -> dict[str, np.ndarray]:
    hand_dir = processed_demo_dir / "hand_processor"
    return {
        "left": np.load(hand_dir / "hand_data_left.npz")["kpts_2d"].astype(np.float32),
        "right": np.load(hand_dir / "hand_data_right.npz")["kpts_2d"].astype(np.float32),
    }


def draw_segment(
    layer: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    radius: int,
    *,
    color: tuple[int, int, int],
    outline: tuple[int, int, int] = (24, 27, 30),
) -> None:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return
    p0 = tuple(np.round(a).astype(int).tolist())
    p1 = tuple(np.round(b).astype(int).tolist())
    cv2.line(layer, p0, p1, outline, radius + 18, cv2.LINE_AA)
    cv2.line(layer, p0, p1, color, radius, cv2.LINE_AA)
    highlight_radius = max(3, radius // 5)
    cv2.line(layer, p0, p1, (238, 241, 241), highlight_radius, cv2.LINE_AA)


def draw_joint(layer: np.ndarray, point: np.ndarray, radius: int) -> None:
    if not np.isfinite(point).all():
        return
    center = tuple(np.round(point).astype(int).tolist())
    cv2.circle(layer, center, radius + 12, (20, 22, 25), -1, cv2.LINE_AA)
    cv2.circle(layer, center, radius, (54, 59, 64), -1, cv2.LINE_AA)
    cv2.circle(layer, center, max(4, radius // 3), (214, 220, 221), -1, cv2.LINE_AA)


def draw_simple_gripper(layer: np.ndarray, hand: np.ndarray, side: str, radius: int) -> None:
    wrist = hand[0]
    thumb = hand[4]
    index = hand[8]
    middle = hand[12]
    if not (np.isfinite(wrist).all() and np.isfinite(thumb).all() and np.isfinite(index).all()):
        return
    virtual_finger = 0.7 * index + 0.3 * middle if np.isfinite(middle).all() else index
    center = 0.5 * (thumb + virtual_finger)
    jaw = thumb - virtual_finger
    norm = np.linalg.norm(jaw)
    if norm < 1e-4:
        return
    jaw = jaw / norm
    approach = center - wrist
    approach_norm = np.linalg.norm(approach)
    if approach_norm < 1e-4:
        approach = np.asarray([0.0, -1.0 if side == "left" else 1.0], dtype=np.float32)
    else:
        approach = approach / approach_norm
    palm = center - approach * radius * 0.55
    jaw_half = min(max(norm * 0.65, radius * 0.8), radius * 2.4)
    for sign in (-1.0, 1.0):
        base = palm + sign * jaw * jaw_half
        tip = center + sign * jaw * jaw_half + approach * radius * 1.55
        cv2.line(
            layer,
            tuple(np.round(base).astype(int).tolist()),
            tuple(np.round(tip).astype(int).tolist()),
            (16, 18, 20),
            max(9, radius // 3),
            cv2.LINE_AA,
        )
        cv2.line(
            layer,
            tuple(np.round(base).astype(int).tolist()),
            tuple(np.round(tip).astype(int).tolist()),
            (60, 64, 68),
            max(5, radius // 5),
            cv2.LINE_AA,
        )
    draw_joint(layer, palm, max(10, radius // 2))


def make_cover_layer(
    shape: tuple[int, int, int],
    arm_points: dict[str, np.ndarray],
    hand_points: dict[str, np.ndarray],
    frame_idx: int,
    link_radius: int,
    hand_radius: int,
) -> np.ndarray:
    layer = np.zeros(shape, dtype=np.uint8)
    colors = {"left": (202, 207, 207), "right": (218, 221, 220)}
    for side in ("left", "right"):
        pts = arm_points[side][frame_idx]
        radii = [int(link_radius * 1.08), int(link_radius), int(link_radius * 0.9)]
        for seg_idx, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            draw_segment(layer, a, b, radii[min(seg_idx, len(radii) - 1)], color=colors[side])
        for point in pts[1:]:
            draw_joint(layer, point, max(12, int(link_radius * 0.42)))
        draw_simple_gripper(layer, hand_points[side][frame_idx], side, hand_radius)
    return layer


def label_frame(frame: np.ndarray, text: str, idx: int) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (185, 48), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"f{idx}", (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def write_montage(
    output_path: Path,
    clean: np.ndarray,
    overlay: np.ndarray,
    final: np.ndarray,
    frame_ids: list[int],
) -> None:
    rows = []
    for name, frames in (("clean", clean), ("robot_overlay", overlay), ("mechanical_cover", final)):
        cells = []
        for idx in frame_ids:
            if idx < 0 or idx >= len(frames):
                continue
            img = cv2.resize(frames[idx], (640, 360), interpolation=cv2.INTER_AREA)
            cells.append(label_frame(img, name, idx))
        if cells:
            rows.append(np.concatenate(cells, axis=1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    mask_path = args.mask_path or processed_demo_dir / "segmentation_processor" / "masks_arm_sam31_d5.npy"
    clean = read_video_rgb(args.clean_video.resolve())
    robot_overlay = read_video_rgb(args.robot_overlay_video.resolve())
    masks = np.load(mask_path.resolve()).astype(np.uint8)
    if len(clean) != len(robot_overlay) or len(clean) != len(masks):
        raise RuntimeError(f"Frame count mismatch: clean={len(clean)} overlay={len(robot_overlay)} masks={len(masks)}")
    height, width = clean.shape[1:3]
    arm_points = load_arm_points(processed_demo_dir, width, height)
    hand_points = load_hand_points(processed_demo_dir)
    scale = height / 1080.0
    link_radius = max(10, int(round(args.link_radius * scale)))
    hand_radius = max(8, int(round(args.hand_radius * scale)))
    kernel_size = max(1, int(round(args.mask_dilation * scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    final_frames = []
    for idx, (clean_frame, robot_frame, mask) in enumerate(zip(clean, robot_overlay, masks)):
        cover_mask = cv2.dilate(mask, kernel, iterations=1).astype(bool)
        robot_diff = np.linalg.norm(robot_frame.astype(np.int16) - clean_frame.astype(np.int16), axis=-1)
        robot_mask = robot_diff > args.robot_diff_threshold
        layer = make_cover_layer(clean_frame.shape, arm_points, hand_points, idx, link_radius, hand_radius)
        layer_mask = (layer.sum(axis=-1) > 0) & cover_mask
        out = clean_frame.copy()
        out[layer_mask] = layer[layer_mask]
        out[robot_mask] = robot_frame[robot_mask]
        final_frames.append(out)

    final = np.asarray(final_frames, dtype=np.uint8)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        args.output_video,
        final,
        fps=args.fps,
        codec="libx264",
        output_params=["-crf", "18", "-pix_fmt", "yuv420p", "-vf", f"scale={width}:{height}"],
    )
    if args.montage:
        write_montage(args.montage, clean, robot_overlay, final, args.montage_frames)
    print(f"mechanical_cover_video={args.output_video}")
    if args.montage:
        print(f"montage={args.montage}")


if __name__ == "__main__":
    main()
