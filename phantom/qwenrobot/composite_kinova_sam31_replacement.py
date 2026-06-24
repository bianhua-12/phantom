from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")
    return np.asarray(frames, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        key = "robot_mask" if "robot_mask" in data.files else data.files[0]
        return data[key].astype(bool)
    return np.load(path, allow_pickle=True).astype(bool)


def resize_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    if masks.shape[1:3] == (height, width):
        return masks.astype(bool)
    out = np.zeros((len(masks), height, width), dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def dilate_masks(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return masks.astype(bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out


def feather_composite(raw: np.ndarray, fill: np.ndarray, masks: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 1:
        out = raw.copy()
        out[masks] = fill[masks]
        return out
    blur = feather if feather % 2 == 1 else feather + 1
    out = np.empty_like(raw)
    for idx, (frame, fill_frame, mask) in enumerate(zip(raw, fill, masks)):
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (blur, blur), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        mixed = frame.astype(np.float32) * (1.0 - alpha) + fill_frame.astype(np.float32) * alpha
        out[idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def temporal_nearest_fill(raw: np.ndarray, exclude_masks: np.ndarray) -> np.ndarray:
    n_frames, height, width = exclude_masks.shape
    fallback = np.median(raw, axis=0).astype(np.uint8)
    max_dist = np.iinfo(np.uint16).max

    prev_rgb = np.empty_like(raw)
    prev_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    last_rgb = fallback.copy()
    last_seen = np.full((height, width), -1, dtype=np.int32)
    for idx in range(n_frames):
        visible = ~exclude_masks[idx]
        last_rgb[visible] = raw[idx][visible]
        last_seen[visible] = idx
        prev_rgb[idx] = last_rgb
        dist = np.where(last_seen >= 0, idx - last_seen, max_dist)
        prev_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    next_rgb = np.empty_like(raw)
    next_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    last_rgb = fallback.copy()
    next_seen = np.full((height, width), -1, dtype=np.int32)
    for idx in range(n_frames - 1, -1, -1):
        visible = ~exclude_masks[idx]
        last_rgb[visible] = raw[idx][visible]
        next_seen[visible] = idx
        next_rgb[idx] = last_rgb
        dist = np.where(next_seen >= 0, next_seen - idx, max_dist)
        next_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    return np.where((prev_dist <= next_dist)[..., None], prev_rgb, next_rgb)


def masked_median_background(raw: np.ndarray, exclude_masks: np.ndarray, row_chunk: int) -> np.ndarray:
    fallback = np.median(raw, axis=0).astype(np.uint8)
    height = raw.shape[1]
    background = np.empty_like(fallback)
    chunk_size = max(1, int(row_chunk))
    for y0 in range(0, height, chunk_size):
        y1 = min(height, y0 + chunk_size)
        values = raw[:, y0:y1].astype(np.float32)
        values = np.where(exclude_masks[:, y0:y1, :, None], np.nan, values)
        median = np.nanmedian(values, axis=0)
        bad = ~np.isfinite(median)
        if np.any(bad):
            median[bad] = fallback[y0:y1][bad]
        background[y0:y1] = np.clip(median, 0, 255).astype(np.uint8)
    return background


def opencv_inpaint_clean(raw: np.ndarray, masks: np.ndarray, radius: float) -> np.ndarray:
    out = np.empty_like(raw)
    for idx, (frame, mask) in enumerate(zip(raw, masks)):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), float(radius), cv2.INPAINT_TELEA)
        out[idx] = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return out


def saturation_detail_masks(frames: np.ndarray, sat_threshold: int, value_threshold: int, dilation: int) -> np.ndarray:
    masks = np.zeros(frames.shape[:3], dtype=bool)
    for idx, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = (hsv[:, :, 1] >= int(sat_threshold)) & (hsv[:, :, 2] >= int(value_threshold))
        if dilation > 1:
            kernel_size = dilation if dilation % 2 == 1 else dilation + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        masks[idx] = mask
    return masks


def label(img: np.ndarray, text: str, idx: int) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (240, 50), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"f{idx}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def color_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


def write_montage(path: Path, rows: list[tuple[str, np.ndarray]], frame_ids: list[int]) -> None:
    rendered = []
    for name, frames in rows:
        cells = []
        for idx in frame_ids:
            if idx < 0 or idx >= len(frames):
                continue
            img = frames[idx]
            if img.ndim == 2:
                img = color_mask(img.astype(bool), (255, 255, 255))
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cells.append(label(img, name, idx))
        if cells:
            rendered.append(np.concatenate(cells, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rendered, axis=0), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite full Kinova render with SAM3.1/Phantom-dilated EgoDex hand masks.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--kinova-dir", type=Path, required=True)
    parser.add_argument("--sam-mask", type=Path, required=True)
    parser.add_argument("--robot-mask", type=Path, default=None)
    parser.add_argument("--robot-video", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--clean-video", type=Path, default=None)
    parser.add_argument("--clean-output", type=Path, default=None)
    parser.add_argument("--montage", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--clean-method", choices=("temporal", "median", "inpaint", "hybrid"), default="temporal")
    parser.add_argument("--fill-dilation", type=int, default=9)
    parser.add_argument("--composite-dilation", type=int, default=3)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--detail-saturation", type=int, default=55)
    parser.add_argument("--detail-value", type=int, default=40)
    parser.add_argument("--detail-dilation", type=int, default=3)
    parser.add_argument("--robot-mask-dilation", type=int, default=1)
    parser.add_argument("--median-row-chunk", type=int, default=32)
    parser.add_argument("--montage-frames", type=int, nargs="*", default=[0, 120, 240, 360, 520, 680, 840, 996])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    kinova_dir = args.kinova_dir.resolve()
    raw = read_video_rgb(processed_demo_dir / "video_rgb_imgs.mkv")
    robot_video = args.robot_video.resolve() if args.robot_video is not None else kinova_dir / "video_robot_Kinova3_camera_ik.mp4"
    robot = read_video_rgb(robot_video)
    sam_masks = load_mask(args.sam_mask.resolve())
    robot_mask_path = args.robot_mask.resolve() if args.robot_mask is not None else kinova_dir / "robot_masks_Kinova3_camera_ik.npz"
    robot_masks = load_mask(robot_mask_path)

    n = min(len(raw), len(robot), len(sam_masks), len(robot_masks))
    raw = raw[:n]
    robot = robot[:n]
    height, width = raw.shape[1:3]
    sam_masks = resize_masks(sam_masks[:n], width, height)
    robot_masks = resize_masks(robot_masks[:n], width, height)

    fill_masks = dilate_masks(sam_masks, args.fill_dilation)
    composite_masks = dilate_masks(sam_masks, args.composite_dilation)
    if args.clean_method == "hybrid":
        if args.clean_video is None:
            raise ValueError("--clean-method hybrid requires --clean-video for detail preservation")
        detail = read_video_rgb(args.clean_video.resolve())[:n]
        if detail.shape[1:3] != raw.shape[1:3]:
            detail = np.asarray([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in detail], dtype=np.uint8)
        fill = opencv_inpaint_clean(raw, fill_masks, args.inpaint_radius)
        clean = feather_composite(raw, fill, composite_masks, args.feather)
        detail_masks = saturation_detail_masks(detail, args.detail_saturation, args.detail_value, args.detail_dilation)
        keep_detail = detail_masks & composite_masks
        clean[keep_detail] = detail[keep_detail]
        clean_source = f"opencv_inpaint_sam31_plus_detail:{args.clean_video.resolve()}"
    elif args.clean_video is None:
        if args.clean_method == "median":
            background = masked_median_background(raw, fill_masks, args.median_row_chunk)
            fill = np.broadcast_to(background[None], raw.shape)
            clean_source = "masked_median_sam31"
        elif args.clean_method == "inpaint":
            fill = opencv_inpaint_clean(raw, fill_masks, args.inpaint_radius)
            clean_source = "opencv_inpaint_sam31"
        else:
            fill = temporal_nearest_fill(raw, fill_masks)
            clean_source = "temporal_nearest_sam31"
        clean = feather_composite(raw, fill, composite_masks, args.feather)
    else:
        clean = read_video_rgb(args.clean_video.resolve())[:n]
        if clean.shape[1:3] != raw.shape[1:3]:
            clean = np.asarray([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in clean], dtype=np.uint8)
        clean_source = str(args.clean_video.resolve())

    robot_masks = dilate_masks(robot_masks, args.robot_mask_dilation)
    final = clean.copy()
    final[robot_masks] = robot[robot_masks]

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        args.output_video,
        final,
        fps=args.fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )
    if args.clean_output is not None:
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            args.clean_output,
            clean,
            fps=args.fps,
            codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )
    if args.montage is not None:
        write_montage(
            args.montage,
            [
                ("raw", raw),
                ("sam31_mask", sam_masks),
                ("clean", clean),
                ("robot_rgb", robot),
                ("robot_mask", robot_masks),
                ("replacement", final),
            ],
            args.montage_frames,
        )

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "kinova_dir": str(kinova_dir),
        "raw_frames": int(len(raw)),
        "output_video": str(args.output_video),
        "clean_output": str(args.clean_output) if args.clean_output else None,
        "montage": str(args.montage) if args.montage else None,
        "sam_mask": str(args.sam_mask.resolve()),
        "robot_video": str(robot_video),
        "robot_mask": str(robot_mask_path),
        "clean_source": clean_source,
        "clean_method": args.clean_method if args.clean_video is None or args.clean_method == "hybrid" else "external_video",
        "sam_mask_mean_area": float(sam_masks.mean()),
        "fill_mask_mean_area": float(fill_masks.mean()),
        "composite_mask_mean_area": float(composite_masks.mean()),
        "robot_mask_mean_area": float(robot_masks.mean()),
        "robot_vs_sam_overlap": float(np.logical_and(robot_masks, sam_masks).sum() / max(int(sam_masks.sum()), 1)),
    }
    summary_path = args.output_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
