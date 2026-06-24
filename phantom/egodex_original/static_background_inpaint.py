from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean EgoDex background by replacing masked regions from a static background estimate.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--background-exclude-dilation", type=int, default=35)
    parser.add_argument("--background-mode", choices=("mean", "percentile", "max_luma", "temporal_nearest"), default="mean")
    parser.add_argument("--background-percentile", type=float, default=85.0)
    parser.add_argument("--composite-dilation", type=int, default=9)
    parser.add_argument("--feather", type=int, default=17)
    return parser.parse_args()


def dilate_masks(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return masks.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out


def estimate_background_mean(frames: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    sums = np.zeros(frames.shape[1:], dtype=np.float64)
    counts = np.zeros(frames.shape[1:3], dtype=np.float64)
    for frame, mask in zip(frames, exclude):
        keep = ~mask
        sums += frame.astype(np.float64) * keep[..., None]
        counts += keep.astype(np.float64)

    median = np.median(frames, axis=0).astype(np.float64)
    bg = median
    valid = counts > 0
    bg[valid] = sums[valid] / counts[valid, None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def estimate_background_percentile(frames: np.ndarray, exclude: np.ndarray, percentile: float) -> np.ndarray:
    channels = []
    fallback_channels = []
    for channel in range(frames.shape[-1]):
        values = frames[..., channel].astype(np.float32)
        fallback_channels.append(np.percentile(values, percentile, axis=0))
        values[exclude] = np.nan
        channels.append(np.nanpercentile(values, percentile, axis=0))

    bg = np.stack(channels, axis=-1)
    fallback = np.stack(fallback_channels, axis=-1)
    bg = np.where(np.isfinite(bg), bg, fallback)
    return np.clip(bg, 0, 255).astype(np.uint8)


def estimate_background_max_luma(frames: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    best_luma = np.full(frames.shape[1:3], -1.0, dtype=np.float32)
    best_rgb = np.zeros(frames.shape[1:], dtype=np.uint8)
    fallback_luma = np.full(frames.shape[1:3], -1.0, dtype=np.float32)
    fallback_rgb = np.zeros(frames.shape[1:], dtype=np.uint8)

    for frame, mask in zip(frames, exclude):
        frame_f = frame.astype(np.float32)
        luma = 0.299 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.114 * frame_f[..., 2]

        fallback_update = luma > fallback_luma
        fallback_luma[fallback_update] = luma[fallback_update]
        fallback_rgb[fallback_update] = frame[fallback_update]

        update = (luma > best_luma) & (~mask)
        best_luma[update] = luma[update]
        best_rgb[update] = frame[update]

    missing = best_luma < 0
    best_rgb[missing] = fallback_rgb[missing]
    return best_rgb


def composite(frames: np.ndarray, background: np.ndarray, masks: np.ndarray, feather: int) -> np.ndarray:
    out = np.empty_like(frames)
    blur_size = max(1, feather)
    if blur_size % 2 == 0:
        blur_size += 1
    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        alpha = mask.astype(np.float32)
        if blur_size > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        mixed = frame.astype(np.float32) * (1.0 - alpha) + background.astype(np.float32) * alpha
        out[idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def temporal_nearest_fill(
    frames: np.ndarray,
    exclude: np.ndarray,
    masks: np.ndarray,
    fallback: np.ndarray,
    feather: int,
) -> np.ndarray:
    n_frames, height, width = masks.shape
    max_dist = np.iinfo(np.uint16).max

    prev_rgb = np.empty_like(frames)
    prev_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    last_rgb = fallback.copy()
    last_seen = np.full((height, width), -1, dtype=np.int32)
    for idx in range(n_frames):
        visible = ~exclude[idx]
        last_rgb[visible] = frames[idx][visible]
        last_seen[visible] = idx
        prev_rgb[idx] = last_rgb
        dist = np.where(last_seen >= 0, idx - last_seen, max_dist)
        prev_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    next_rgb = np.empty_like(frames)
    next_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    next_seen = np.full((height, width), -1, dtype=np.int32)
    last_rgb = fallback.copy()
    for idx in range(n_frames - 1, -1, -1):
        visible = ~exclude[idx]
        last_rgb[visible] = frames[idx][visible]
        next_seen[visible] = idx
        next_rgb[idx] = last_rgb
        dist = np.where(next_seen >= 0, next_seen - idx, max_dist)
        next_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    use_prev = prev_dist <= next_dist
    fill = np.where(use_prev[..., None], prev_rgb, next_rgb)

    out = np.empty_like(frames)
    blur_size = max(1, feather)
    if blur_size % 2 == 0:
        blur_size += 1
    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        alpha = mask.astype(np.float32)
        if blur_size > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        mixed = frame.astype(np.float32) * (1.0 - alpha) + fill[idx].astype(np.float32) * alpha
        out[idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def write_montage(processed_demo_dir: Path, frames: np.ndarray, masks: np.ndarray, output_frames: np.ndarray) -> Path:
    ids = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int).tolist()
    rows = []
    for label, source in (("raw", frames), ("mask", masks), ("static_clean", output_frames)):
        cells = []
        for idx in ids:
            if source.ndim == 3:
                img = cv2.cvtColor((source[idx].astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
            else:
                img = source[idx].copy()
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(idx), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.concatenate(cells, axis=1))
    out = processed_demo_dir / "inpaint_processor" / "static_background_inpaint_montage.jpg"
    iio.imwrite(out, np.concatenate(rows, axis=0), quality=92)
    return out


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    mask_path = args.mask_path.resolve()
    output_video = args.output_video.resolve()

    frames = media.read_video(processed_demo_dir / "video_rgb_imgs.mkv").astype(np.uint8)
    masks = np.load(mask_path).astype(bool)
    if len(frames) != len(masks):
        raise ValueError(f"Frame/mask length mismatch: {len(frames)} vs {len(masks)}")

    exclude = dilate_masks(masks, args.background_exclude_dilation)
    if args.background_mode == "percentile":
        background = estimate_background_percentile(frames, exclude, args.background_percentile)
    elif args.background_mode == "max_luma":
        background = estimate_background_max_luma(frames, exclude)
    else:
        background = estimate_background_mean(frames, exclude)
    composite_masks = dilate_masks(masks, args.composite_dilation)
    if args.background_mode == "temporal_nearest":
        clean = temporal_nearest_fill(frames, exclude, composite_masks, background, args.feather)
    else:
        clean = composite(frames, background, composite_masks, args.feather)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output_video, clean, fps=args.fps, codec="ffv1")
    iio.imwrite(output_video.parent / "static_background_estimate.jpg", background, quality=92)
    montage = write_montage(processed_demo_dir, frames, composite_masks, clean)

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "mask_path": str(mask_path),
        "output_video": str(output_video),
        "background_exclude_dilation": int(args.background_exclude_dilation),
        "background_mode": args.background_mode,
        "background_percentile": float(args.background_percentile),
        "composite_dilation": int(args.composite_dilation),
        "feather": int(args.feather),
        "frames": int(len(frames)),
        "mask_mean_area": float(masks.mean()),
        "composite_mask_mean_area": float(composite_masks.mean()),
        "montage": str(montage),
    }
    summary_path = output_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
