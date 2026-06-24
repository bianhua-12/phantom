from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np
import requests
import torch
from sam2.build_sam import build_sam2_video_predictor


SAM2_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine Phantom EgoDex arm masks with the original SAM2 video predictor.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("submodules/sam2/checkpoints/sam2_hiera_large.pt"))
    parser.add_argument("--model-cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--max-anchors", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=21)
    parser.add_argument("--close-kernel", type=int, default=11)
    parser.add_argument("--post-dilation", type=int, default=3)
    parser.add_argument("--replace", action="store_true", help="Replace segmentation_processor/masks_arm.npy after saving a .rough.npy backup.")
    return parser.parse_args()


def download_checkpoint(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(SAM2_CKPT_URL, stream=True, timeout=30) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def choose_anchors(masks: np.ndarray, max_anchors: int) -> list[int]:
    nonempty = np.where(masks.reshape(len(masks), -1).sum(axis=1) > 0)[0]
    if len(nonempty) == 0:
        raise ValueError("No nonempty masks to seed SAM2")
    if max_anchors <= 1:
        return [int(nonempty[len(nonempty) // 2])]
    positions = np.linspace(0, len(nonempty) - 1, min(max_anchors, len(nonempty)), dtype=int)
    return [int(nonempty[i]) for i in positions]


def dilate_masks(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return masks.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out


def propagate_from_anchor(predictor, frames_dir: Path, anchor_idx: int, seed_mask: np.ndarray, n_frames: int) -> dict[int, np.ndarray]:
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.reset_state(state)
    predictor.add_new_mask(state, frame_idx=anchor_idx, obj_id=0, mask=seed_mask.astype(bool))

    out: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        for frame_idx, _obj_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=anchor_idx,
            max_frame_num_to_track=n_frames - anchor_idx,
            reverse=False,
        ):
            out[int(frame_idx)] = (logits[0, 0].detach().cpu().numpy() > 0.0)
        for frame_idx, _obj_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=anchor_idx,
            max_frame_num_to_track=anchor_idx + 1,
            reverse=True,
        ):
            out[int(frame_idx)] = (logits[0, 0].detach().cpu().numpy() > 0.0)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def merge_candidates(raw_masks: np.ndarray, candidates: list[dict[int, np.ndarray]], clip_masks: np.ndarray) -> np.ndarray:
    refined = np.zeros_like(raw_masks, dtype=bool)
    for frame_idx in range(len(raw_masks)):
        best_mask = None
        best_score = -1.0
        for candidate in candidates:
            if frame_idx not in candidate:
                continue
            mask = np.logical_and(candidate[frame_idx], clip_masks[frame_idx])
            score = iou(mask, raw_masks[frame_idx])
            if score > best_score:
                best_score = score
                best_mask = mask
        refined[frame_idx] = best_mask if best_mask is not None and best_mask.any() else raw_masks[frame_idx]
    return refined


def postprocess_masks(masks: np.ndarray, close_kernel: int, post_dilation: int) -> np.ndarray:
    close_kernel_arr = np.ones((close_kernel, close_kernel), np.uint8) if close_kernel > 1 else None
    dilate_kernel_arr = np.ones((post_dilation, post_dilation), np.uint8) if post_dilation > 1 else None
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        current = mask.astype(np.uint8)
        if close_kernel_arr is not None:
            current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, close_kernel_arr, iterations=1)
        if dilate_kernel_arr is not None:
            current = cv2.dilate(current, dilate_kernel_arr, iterations=1)
        out[idx] = current.astype(bool)
    return out


def write_visualizations(processed_demo_dir: Path, masks: np.ndarray) -> tuple[Path, Path, Path]:
    frames = media.read_video(processed_demo_dir / "video_L.mp4")
    mask_video = (masks.astype(np.uint8) * 255)
    overlay = frames.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[masks] = (0.55 * overlay[masks] + 0.45 * red[masks]).astype(np.uint8)
    blacked = frames.copy()
    blacked[masks] = 0

    out_dir = processed_demo_dir / "segmentation_processor"
    mask_path = out_dir / "video_masks_arm_sam2.mkv"
    overlay_path = out_dir / "video_sam_arm_sam2_overlay.mkv"
    blacked_path = out_dir / "video_sam_arm_sam2.mkv"
    media.write_video(mask_path, mask_video, fps=15, codec="ffv1")
    media.write_video(overlay_path, overlay, fps=15, codec="ffv1")
    media.write_video(blacked_path, blacked, fps=15, codec="ffv1")
    return mask_path, overlay_path, blacked_path


def write_montage(processed_demo_dir: Path, raw_masks: np.ndarray, refined_masks: np.ndarray) -> Path:
    frames = media.read_video(processed_demo_dir / "video_L.mp4")
    ids = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    rows = []
    for label, source in (("raw", frames), ("rough_mask", raw_masks), ("sam2_mask", refined_masks)):
        row = []
        for idx in ids:
            if source.ndim == 3:
                img = cv2.cvtColor((source[idx].astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
            else:
                img = source[idx].copy()
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(int(idx)), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
    montage = np.concatenate(rows, axis=0)
    path = processed_demo_dir / "segmentation_processor" / "sam2_mask_refine_montage.jpg"
    iio.imwrite(path, montage, quality=92)
    return path


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    frames_dir = processed_demo_dir / "original_images"
    masks_path = processed_demo_dir / "segmentation_processor" / "masks_arm.npy"
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    if not masks_path.exists():
        raise FileNotFoundError(masks_path)

    raw_masks = np.load(masks_path).astype(bool)
    anchors = choose_anchors(raw_masks, args.max_anchors)
    download_checkpoint(args.checkpoint)

    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device="cuda" if torch.cuda.is_available() else "cpu")
    candidates = [propagate_from_anchor(predictor, frames_dir, anchor, raw_masks[anchor], len(raw_masks)) for anchor in anchors]
    refined = merge_candidates(raw_masks, candidates, dilate_masks(raw_masks, args.dilation))
    refined = postprocess_masks(refined, args.close_kernel, args.post_dilation)

    out_path = processed_demo_dir / "segmentation_processor" / "masks_arm_sam2.npy"
    np.save(out_path, refined.astype(np.uint8))
    mask_video, overlay_video, blacked_video = write_visualizations(processed_demo_dir, refined)
    montage = write_montage(processed_demo_dir, raw_masks, refined)

    if args.replace:
        backup = masks_path.with_name("masks_arm.rough.npy")
        if not backup.exists():
            np.save(backup, raw_masks.astype(np.uint8))
        np.save(masks_path, refined.astype(np.uint8))

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "anchors": anchors,
        "rough_mask_mean_area": float(raw_masks.mean()),
        "sam2_mask_mean_area": float(refined.mean()),
        "close_kernel": int(args.close_kernel),
        "post_dilation": int(args.post_dilation),
        "masks_arm_sam2": str(out_path),
        "mask_video": str(mask_video),
        "overlay_video": str(overlay_video),
        "blacked_video": str(blacked_video),
        "montage": str(montage),
        "replaced_masks_arm": bool(args.replace),
    }
    summary_path = processed_demo_dir / "segmentation_processor" / "sam2_mask_refine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
