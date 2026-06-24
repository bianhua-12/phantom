from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from phantom.egodex_original.sam31_segment import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MONTAGE_FRAMES,
    frame_paths,
    postprocess,
)


@dataclass
class SeedPrompt:
    side: str
    frame_index: int
    bbox_xyxy: list[float]
    bbox_xywh_norm: list[float]
    margin_to_edge: float
    detected_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EgoDex masks with SAM3.1 using the Phantom-style "
            "high-quality seed frame + hand keypoint video propagation setup."
        )
    )
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--left-hand-data", type=Path, default=None)
    parser.add_argument("--right-hand-data", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="masks_arm_sam31_keypoint.npy")
    parser.add_argument(
        "--prompt-mode",
        choices=("box_then_points", "points"),
        default="box_then_points",
        help="box_then_points is closest to Phantom; points avoids SAM3.1 box refinement.",
    )
    parser.add_argument("--bbox-padding", type=float, default=20.0)
    parser.add_argument("--post-dilation", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=0)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--compare-mask", type=Path, default=None)
    parser.add_argument("--montage-frames", type=int, nargs="*", default=list(DEFAULT_MONTAGE_FRAMES))
    return parser.parse_args()


def keypoint_bboxes(kpts_2d: np.ndarray, width: int, height: int, padding: float) -> np.ndarray:
    mins = np.nanmin(kpts_2d, axis=1)
    maxs = np.nanmax(kpts_2d, axis=1)
    x0 = np.clip(mins[:, 0] - padding, 0, width - 1)
    y0 = np.clip(mins[:, 1] - padding, 0, height - 1)
    x1 = np.clip(maxs[:, 0] + padding, 0, width - 1)
    y1 = np.clip(maxs[:, 1] + padding, 0, height - 1)
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)


def choose_seed(hand_data: dict[str, np.ndarray], side: str, width: int, height: int, padding: float) -> SeedPrompt:
    detected = hand_data["hand_detected"].astype(bool)
    kpts_2d = hand_data["kpts_2d"].astype(np.float32)
    bboxes = keypoint_bboxes(kpts_2d, width, height, padding)
    valid_shape = (bboxes[:, 2] > bboxes[:, 0] + 2) & (bboxes[:, 3] > bboxes[:, 1] + 2)
    valid_kpts = np.isfinite(kpts_2d).all(axis=(1, 2))
    valid = detected & valid_shape & valid_kpts
    if not valid.any():
        raise RuntimeError(f"No valid {side} hand prompt frames found")

    margins = np.minimum.reduce([bboxes[:, 0], bboxes[:, 1], width - 1 - bboxes[:, 2], height - 1 - bboxes[:, 3]])
    margins[~valid] = -np.inf
    frame_index = int(np.argmax(margins))
    x0, y0, x1, y1 = bboxes[frame_index].astype(float).tolist()
    bbox_xywh_norm = [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]
    return SeedPrompt(
        side=side,
        frame_index=frame_index,
        bbox_xyxy=[x0, y0, x1, y1],
        bbox_xywh_norm=bbox_xywh_norm,
        margin_to_edge=float(margins[frame_index]),
        detected_frames=int(detected.sum()),
    )


def prompt_points(hand_data: dict[str, np.ndarray], frame_index: int, width: int, height: int) -> np.ndarray:
    points = hand_data["kpts_2d"][frame_index].astype(np.float32).copy()
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    points[:, 0] /= float(width)
    points[:, 1] /= float(height)
    return points


def masks_from_outputs(outputs: dict[str, Any], height: int, width: int) -> dict[int, np.ndarray]:
    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks")
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.detach().cpu().numpy()

    masks: dict[int, np.ndarray] = {}
    if binary_masks is None:
        return masks
    for idx, obj_id in enumerate(obj_ids):
        mask = binary_masks[idx]
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[int(obj_id)] = mask
    return masks


def first_obj_id(response: dict[str, Any]) -> int | None:
    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    if len(obj_ids) == 0:
        return None
    return int(obj_ids[0])


def add_hand_prompt(
    model: Any,
    session_id: str,
    seed: SeedPrompt,
    points_norm: np.ndarray,
    prompt_mode: str,
    requested_obj_id: int,
) -> tuple[int, str]:
    point_labels = np.ones(len(points_norm), dtype=np.int32)
    if prompt_mode == "points":
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed.frame_index,
                "points": points_norm,
                "point_labels": point_labels,
                "obj_id": requested_obj_id,
            }
        )
        return requested_obj_id, "points"

    try:
        box_response = model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed.frame_index,
                "bounding_boxes": np.asarray([seed.bbox_xywh_norm], dtype=np.float32),
                "bounding_box_labels": np.ones(1, dtype=np.int32),
            }
        )
    except Exception as exc:
        raise RuntimeError(
            "SAM3.1 box prompt failed in box_then_points mode. "
            "Use --prompt-mode points to run the explicit points-only path."
        ) from exc

    obj_id = first_obj_id(box_response)
    if obj_id is None:
        obj_id = requested_obj_id
    model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": seed.frame_index,
            "points": points_norm,
            "point_labels": point_labels,
            "obj_id": obj_id,
        }
    )
    return obj_id, "box_then_points"


def run_sam31_keypoints(
    frame_dir: Path,
    checkpoint: Path,
    left_data: dict[str, np.ndarray],
    right_data: dict[str, np.ndarray],
    left_seed: SeedPrompt,
    right_seed: SeedPrompt,
    width: int,
    height: int,
    prompt_mode: str,
    max_num_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache_sam31_egodex")
    from sam3 import build_sam3_predictor

    frame_count = len(frame_paths(frame_dir))

    def propagate_one_hand(model: Any, hand_data: dict[str, np.ndarray], seed: SeedPrompt) -> tuple[np.ndarray, dict[str, Any]]:
        response = model.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
        session_id = response["session_id"]
        points = prompt_points(hand_data, seed.frame_index, width, height)
        obj_id, mode = add_hand_prompt(model, session_id, seed, points, prompt_mode, requested_obj_id=0)

        masks = np.zeros((frame_count, height, width), dtype=np.uint8)
        objects_per_frame: dict[str, int] = {}
        for response in model.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            frame_idx = response.get("frame_index")
            if frame_idx is None:
                continue
            frame_masks = masks_from_outputs(response.get("outputs", {}), height, width)
            objects_per_frame[str(int(frame_idx))] = len(frame_masks)
            if obj_id in frame_masks:
                masks[int(frame_idx)] = frame_masks[obj_id]
        return masks, {
            "session_id": session_id,
            "obj_id": int(obj_id),
            "prompt_mode": mode,
            "objects_per_frame": objects_per_frame,
        }

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model = build_sam3_predictor(
            checkpoint_path=str(checkpoint),
            version="sam3.1",
            compile=False,
            warm_up=False,
            use_fa3=False,
            async_loading_frames=False,
            max_num_objects=max_num_objects,
            multiplex_count=max_num_objects,
        )
        left_masks, left_info = propagate_one_hand(model, left_data, left_seed)
        right_masks, right_info = propagate_one_hand(model, right_data, right_seed)
        torch.cuda.synchronize()

    union = (left_masks | right_masks).astype(np.uint8)
    prompt_info = {
        "independent_sessions": True,
        "left": left_info,
        "right": right_info,
    }
    return left_masks, right_masks, union, prompt_info


def labeled_cell(image_rgb: np.ndarray, label: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 230), 22), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def color_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    height, width = mask.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[mask.astype(bool)] = color
    return out


def color_diff_mask(text_mask: np.ndarray, keypoint_mask: np.ndarray) -> np.ndarray:
    text = text_mask.astype(bool)
    keypoint = keypoint_mask.astype(bool)
    out = np.zeros((*text.shape, 3), dtype=np.uint8)
    out[text & ~keypoint] = (255, 0, 0)
    out[keypoint & ~text] = (0, 255, 0)
    out[text & keypoint] = (255, 220, 0)
    return out


def write_compare_montage(
    paths: list[Path],
    left_masks: np.ndarray,
    right_masks: np.ndarray,
    keypoint_masks: np.ndarray,
    text_masks: np.ndarray | None,
    frames: list[int],
    seeds: list[SeedPrompt],
    output_path: Path,
) -> None:
    valid_frames = [idx for idx in frames if 0 <= idx < len(paths)]
    seed_frames = [seed.frame_index for seed in seeds]
    valid_frames = sorted(dict.fromkeys(valid_frames + seed_frames))
    if not valid_frames:
        valid_frames = np.linspace(0, len(paths) - 1, min(6, len(paths)), dtype=int).tolist()

    columns: list[np.ndarray] = []
    for idx in valid_frames:
        rows: list[np.ndarray] = []
        if text_masks is not None:
            rows.append(labeled_cell(color_mask(text_masks[idx], (255, 0, 0)), f"f{idx} text mask area={text_masks[idx].mean():.3f}"))
        rows.append(labeled_cell(color_mask(left_masks[idx], (0, 220, 255)), f"left keypoint area={left_masks[idx].mean():.3f}"))
        rows.append(labeled_cell(color_mask(right_masks[idx], (80, 140, 255)), f"right keypoint area={right_masks[idx].mean():.3f}"))
        rows.append(labeled_cell(color_mask(keypoint_masks[idx], (0, 255, 0)), f"union keypoint area={keypoint_masks[idx].mean():.3f}"))
        if text_masks is not None:
            rows.append(labeled_cell(color_diff_mask(text_masks[idx], keypoint_masks[idx]), "diff red=text green=key yellow=overlap"))
        columns.append(np.concatenate(rows, axis=0))

    montage = np.concatenate(columns, axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def load_hand_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum(axis=(1, 2)).astype(np.float64)
    union = np.logical_or(a, b).sum(axis=(1, 2)).astype(np.float64)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    frame_dir = args.frame_dir.resolve() if args.frame_dir is not None else processed_demo_dir / "original_images"
    left_hand_path = args.left_hand_data.resolve() if args.left_hand_data is not None else processed_demo_dir / "hand_processor" / "hand_data_left.npz"
    right_hand_path = args.right_hand_data.resolve() if args.right_hand_data is not None else processed_demo_dir / "hand_processor" / "hand_data_right.npz"
    paths = frame_paths(frame_dir)

    first_frame = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Cannot read frame: {paths[0]}")
    height, width = first_frame.shape[:2]

    left_data = load_hand_npz(left_hand_path)
    right_data = load_hand_npz(right_hand_path)
    left_seed = choose_seed(left_data, "left", width, height, args.bbox_padding)
    right_seed = choose_seed(right_data, "right", width, height, args.bbox_padding)

    left_masks, right_masks, masks, prompt_info = run_sam31_keypoints(
        frame_dir=frame_dir,
        checkpoint=args.checkpoint.resolve(),
        left_data=left_data,
        right_data=right_data,
        left_seed=left_seed,
        right_seed=right_seed,
        width=width,
        height=height,
        prompt_mode=args.prompt_mode,
        max_num_objects=args.max_num_objects,
    )
    masks = np.stack([postprocess(mask, args.close_kernel, args.post_dilation) for mask in masks], axis=0)

    segmentation_dir = processed_demo_dir / "segmentation_processor"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    output_path = segmentation_dir / args.output_name
    np.save(output_path, masks.astype(np.uint8))
    stem = Path(args.output_name).stem
    left_output_path = segmentation_dir / f"{stem}_left.npy"
    right_output_path = segmentation_dir / f"{stem}_right.npy"
    np.save(left_output_path, left_masks.astype(np.uint8))
    np.save(right_output_path, right_masks.astype(np.uint8))

    text_masks = None
    compare_mask_path = args.compare_mask
    if compare_mask_path is None:
        default_compare = segmentation_dir / "masks_arm_sam31_strict.npy"
        compare_mask_path = default_compare if default_compare.exists() else None
    if compare_mask_path is not None and compare_mask_path.exists():
        text_masks = np.load(compare_mask_path).astype(np.uint8)
        if text_masks.shape != masks.shape:
            raise RuntimeError(f"Compare mask shape {text_masks.shape} does not match {masks.shape}: {compare_mask_path}")

    montage_path = segmentation_dir / f"{stem}_compare.jpg"
    write_compare_montage(
        paths=paths,
        left_masks=left_masks,
        right_masks=right_masks,
        keypoint_masks=masks,
        text_masks=text_masks,
        frames=args.montage_frames,
        seeds=[left_seed, right_seed],
        output_path=montage_path,
    )

    summary: dict[str, Any] = {
        "processed_demo_dir": str(processed_demo_dir),
        "frame_dir": str(frame_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_mode_requested": args.prompt_mode,
        "prompt_info": prompt_info,
        "frames": int(len(paths)),
        "image_size": [int(width), int(height)],
        "bbox_padding": float(args.bbox_padding),
        "close_kernel": int(args.close_kernel),
        "post_dilation": int(args.post_dilation),
        "left_seed": asdict(left_seed),
        "right_seed": asdict(right_seed),
        "output": str(output_path),
        "left_output": str(left_output_path),
        "right_output": str(right_output_path),
        "montage": str(montage_path),
        "mean_area": float(masks.mean()),
        "min_area": float(masks.mean(axis=(1, 2)).min()),
        "max_area": float(masks.mean(axis=(1, 2)).max()),
    }
    if text_masks is not None:
        iou = compute_iou(masks, text_masks)
        summary.update(
            {
                "compare_mask": str(compare_mask_path),
                "compare_mean_area": float(text_masks.mean()),
                "compare_mean_iou": float(iou.mean()),
                "compare_min_iou": float(iou.min()),
                "compare_max_iou": float(iou.max()),
            }
        )

    summary_path = segmentation_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
