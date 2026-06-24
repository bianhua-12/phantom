from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand EgoDex arm masks to include nearby arm shadows.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--input-mask", type=Path, default=None)
    parser.add_argument("--output-mask-name", type=str, default="masks_arm_shadow.npy")
    parser.add_argument("--background-exclude-dilation", type=int, default=35)
    parser.add_argument("--shadow-search-dilation", type=int, default=65)
    parser.add_argument("--shadow-delta", type=float, default=18.0)
    parser.add_argument("--color-delta", type=float, default=8.0)
    parser.add_argument("--close-kernel", type=int, default=11)
    parser.add_argument("--post-dilation", type=int, default=5)
    return parser.parse_args()


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = np.zeros_like(mask, dtype=bool)
    for idx, frame_mask in enumerate(mask):
        out[idx] = cv2.dilate(frame_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out


def postprocess(mask: np.ndarray, close_kernel: int, post_dilation: int) -> np.ndarray:
    close = np.ones((close_kernel, close_kernel), dtype=np.uint8) if close_kernel > 1 else None
    dilate = np.ones((post_dilation, post_dilation), dtype=np.uint8) if post_dilation > 1 else None
    out = np.zeros_like(mask, dtype=bool)
    for idx, frame_mask in enumerate(mask):
        current = frame_mask.astype(np.uint8)
        if close is not None:
            current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, close, iterations=1)
        if dilate is not None:
            current = cv2.dilate(current, dilate, iterations=1)
        out[idx] = current.astype(bool)
    return out


def estimate_background(frames: np.ndarray, exclude_masks: np.ndarray) -> np.ndarray:
    sums = np.zeros(frames.shape[1:], dtype=np.float64)
    counts = np.zeros(frames.shape[1:3], dtype=np.float64)
    for frame, mask in zip(frames, exclude_masks):
        keep = ~mask
        sums += frame.astype(np.float64) * keep[..., None]
        counts += keep.astype(np.float64)

    median = np.median(frames, axis=0).astype(np.float64)
    background = median
    valid = counts > 0
    background[valid] = sums[valid] / counts[valid, None]
    return np.clip(background, 0, 255).astype(np.uint8)


def luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def make_shadow_mask(
    frames: np.ndarray,
    base_masks: np.ndarray,
    background: np.ndarray,
    search_masks: np.ndarray,
    *,
    shadow_delta: float,
    color_delta: float,
) -> np.ndarray:
    bg_luma = luminance(background)
    out = np.zeros_like(base_masks, dtype=bool)
    for idx, frame in enumerate(frames):
        luma_drop = bg_luma - luminance(frame)
        color_diff = np.abs(background.astype(np.float32) - frame.astype(np.float32)).mean(axis=2)
        shadow = (luma_drop >= shadow_delta) & (color_diff >= color_delta)
        out[idx] = search_masks[idx] & shadow & ~base_masks[idx]
    return out


def write_montage(processed_demo_dir: Path, frames: np.ndarray, base: np.ndarray, shadow: np.ndarray, enhanced: np.ndarray) -> Path:
    ids = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int).tolist()
    rows = []
    for label, source in (
        ("raw", frames),
        ("base", base),
        ("shadow", shadow),
        ("enhanced", enhanced),
    ):
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
    out = processed_demo_dir / "segmentation_processor" / "arm_shadow_mask_montage.jpg"
    iio.imwrite(out, np.concatenate(rows, axis=0), quality=92)
    return out


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    input_mask = (
        args.input_mask.resolve()
        if args.input_mask is not None
        else processed_demo_dir / "segmentation_processor" / "masks_arm.npy"
    )
    if not input_mask.exists():
        raise FileNotFoundError(input_mask)

    frames = media.read_video(processed_demo_dir / "video_L.mp4").astype(np.uint8)
    base = np.load(input_mask).astype(bool)
    if len(frames) != len(base):
        raise ValueError(f"Frame/mask length mismatch: {len(frames)} vs {len(base)}")

    background_exclude = dilate_mask(base, args.background_exclude_dilation)
    background = estimate_background(frames, background_exclude)
    search = dilate_mask(base, args.shadow_search_dilation)
    shadow = make_shadow_mask(
        frames,
        base,
        background,
        search,
        shadow_delta=args.shadow_delta,
        color_delta=args.color_delta,
    )
    enhanced = postprocess(base | shadow, args.close_kernel, args.post_dilation)

    seg_dir = processed_demo_dir / "segmentation_processor"
    output_mask = seg_dir / args.output_mask_name
    np.save(output_mask, enhanced.astype(np.uint8))
    media.write_video(seg_dir / output_mask.with_suffix(".mkv").name, enhanced.astype(np.uint8) * 255, fps=15, codec="ffv1")
    montage = write_montage(processed_demo_dir, frames, base, shadow, enhanced)
    iio.imwrite(seg_dir / "arm_shadow_background.jpg", background, quality=92)

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "input_mask": str(input_mask),
        "output_mask": str(output_mask),
        "base_mean_area": float(base.mean()),
        "shadow_mean_area": float(shadow.mean()),
        "enhanced_mean_area": float(enhanced.mean()),
        "background_exclude_dilation": int(args.background_exclude_dilation),
        "shadow_search_dilation": int(args.shadow_search_dilation),
        "shadow_delta": float(args.shadow_delta),
        "color_delta": float(args.color_delta),
        "close_kernel": int(args.close_kernel),
        "post_dilation": int(args.post_dilation),
        "montage": str(montage),
    }
    summary_path = seg_dir / output_mask.with_suffix(".summary.json").name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
