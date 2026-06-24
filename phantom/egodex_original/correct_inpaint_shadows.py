from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np

from phantom.egodex_original.static_background_inpaint import (
    dilate_masks,
    estimate_background_max_luma,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove dark residual arm shadows from an inpainted EgoDex video.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--inpaint-video", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--background-exclude-dilation", type=int, default=35)
    parser.add_argument("--search-dilation", type=int, default=70)
    parser.add_argument("--dark-delta", type=float, default=18.0)
    parser.add_argument("--color-delta", type=float, default=8.0)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--post-dilation", type=int, default=3)
    parser.add_argument("--feather", type=int, default=17)
    return parser.parse_args()


def luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def postprocess_masks(masks: np.ndarray, close_kernel: int, post_dilation: int) -> np.ndarray:
    close = np.ones((close_kernel, close_kernel), dtype=np.uint8) if close_kernel > 1 else None
    dilate = np.ones((post_dilation, post_dilation), dtype=np.uint8) if post_dilation > 1 else None
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        current = mask.astype(np.uint8)
        if close is not None:
            current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, close, iterations=1)
        if dilate is not None:
            current = cv2.dilate(current, dilate, iterations=1)
        out[idx] = current.astype(bool)
    return out


def build_residual_masks(
    inpainted: np.ndarray,
    background: np.ndarray,
    search_masks: np.ndarray,
    dark_delta: float,
    color_delta: float,
) -> np.ndarray:
    bg_luma = luminance(background)
    masks = np.zeros(search_masks.shape, dtype=bool)
    for idx, frame in enumerate(inpainted):
        luma_drop = bg_luma - luminance(frame)
        color_diff = np.abs(background.astype(np.float32) - frame.astype(np.float32)).mean(axis=2)
        masks[idx] = search_masks[idx] & (luma_drop >= dark_delta) & (color_diff >= color_delta)
    return masks


def composite(inpainted: np.ndarray, background: np.ndarray, masks: np.ndarray, feather: int) -> np.ndarray:
    out = np.empty_like(inpainted)
    blur_size = max(1, feather)
    if blur_size % 2 == 0:
        blur_size += 1
    for idx, (frame, mask) in enumerate(zip(inpainted, masks)):
        alpha = mask.astype(np.float32)
        if blur_size > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        mixed = frame.astype(np.float32) * (1.0 - alpha) + background.astype(np.float32) * alpha
        out[idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def write_montage(processed_demo_dir: Path, raw: np.ndarray, inpainted: np.ndarray, masks: np.ndarray, corrected: np.ndarray) -> Path:
    ids = np.linspace(0, len(raw) - 1, min(8, len(raw)), dtype=int).tolist()
    rows = []
    for label, source in (("raw", raw), ("input_inpaint", inpainted), ("residual_mask", masks), ("corrected", corrected)):
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
    out = processed_demo_dir / "inpaint_processor" / "correct_inpaint_shadows_montage.jpg"
    iio.imwrite(out, np.concatenate(rows, axis=0), quality=92)
    return out


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    raw = media.read_video(processed_demo_dir / "video_rgb_imgs.mkv").astype(np.uint8)
    inpainted = media.read_video(args.inpaint_video.resolve()).astype(np.uint8)
    masks = np.load(args.mask_path.resolve()).astype(bool)
    if len(raw) != len(inpainted) or len(raw) != len(masks):
        raise ValueError(f"Length mismatch: raw={len(raw)}, inpainted={len(inpainted)}, masks={len(masks)}")

    background = estimate_background_max_luma(raw, dilate_masks(masks, args.background_exclude_dilation))
    search = dilate_masks(masks, args.search_dilation)
    residual = build_residual_masks(inpainted, background, search, args.dark_delta, args.color_delta)
    residual = postprocess_masks(residual, args.close_kernel, args.post_dilation)
    corrected = composite(inpainted, background, residual, args.feather)

    output_video = args.output_video.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output_video, corrected, fps=args.fps, codec="ffv1")
    media.write_video(output_video.parent / output_video.with_suffix(".residual_mask.mkv").name, residual.astype(np.uint8) * 255, fps=args.fps, codec="ffv1")
    montage = write_montage(processed_demo_dir, raw, inpainted, residual, corrected)

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "inpaint_video": str(args.inpaint_video.resolve()),
        "mask_path": str(args.mask_path.resolve()),
        "output_video": str(output_video),
        "background_exclude_dilation": int(args.background_exclude_dilation),
        "search_dilation": int(args.search_dilation),
        "dark_delta": float(args.dark_delta),
        "color_delta": float(args.color_delta),
        "residual_mean_area": float(residual.mean()),
        "montage": str(montage),
    }
    summary_path = output_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
