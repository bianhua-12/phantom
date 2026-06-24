from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_CHECKPOINT = Path(
    "/mnt/project_rlinf/jlchen/code/sam3_reference/checkpoints/modelscope_sam3.1/sam3.1_multiplex.pt"
)
DEFAULT_PROMPT = "human hands and arms"
DEFAULT_MONTAGE_FRAMES = (0, 120, 240, 360, 520, 680, 840, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EgoDex human hand/arm masks with SAM3.1 text-prompt video segmentation."
    )
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="masks_arm_sam31.npy")
    parser.add_argument("--replace", action="store_true", help="Replace segmentation_processor/masks_arm.npy.")
    parser.add_argument("--post-dilation", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=0)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--montage-frames", type=int, nargs="*", default=list(DEFAULT_MONTAGE_FRAMES))
    return parser.parse_args()


def frame_paths(frame_dir: Path) -> list[Path]:
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")
    return paths


def postprocess(mask: np.ndarray, close_kernel: int, post_dilation: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if close_kernel > 0:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if post_dilation > 0:
        if post_dilation % 2 == 0:
            post_dilation += 1
        kernel = np.ones((post_dilation, post_dilation), np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)
    return (out > 0).astype(np.uint8)


def run_sam31(
    frame_dir: Path,
    checkpoint: Path,
    prompt: str,
    prompt_frame: int,
    max_num_objects: int,
) -> dict[int, dict[int, np.ndarray]]:
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache_sam31_egodex")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    from sam3 import build_sam3_predictor

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
    response = model.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
    session_id = response["session_id"]
    model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame,
            "text": prompt,
        }
    )

    masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
    for response in model.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        frame_idx = response.get("frame_index")
        if frame_idx is None:
            continue
        outputs = response.get("outputs", {})
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.detach().cpu().numpy()

        frame_masks: dict[int, np.ndarray] = {}
        if binary_masks is not None:
            for idx, obj_id in enumerate(obj_ids):
                mask = binary_masks[idx]
                if mask.ndim == 3:
                    mask = mask[0]
                frame_masks[int(obj_id)] = (mask > 0).astype(np.uint8)
        masks_by_frame[int(frame_idx)] = frame_masks
    torch.cuda.synchronize()
    return masks_by_frame


def make_union_masks(
    paths: list[Path],
    masks_by_frame: dict[int, dict[int, np.ndarray]],
    close_kernel: int,
    post_dilation: int,
) -> np.ndarray:
    union_masks: list[np.ndarray] = []
    for frame_idx, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read frame: {path}")
        height, width = image.shape[:2]
        union = np.zeros((height, width), dtype=np.uint8)
        for mask in masks_by_frame.get(frame_idx, {}).values():
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            union |= mask.astype(np.uint8)
        union_masks.append(postprocess(union, close_kernel, post_dilation))
    return np.stack(union_masks)


def write_montage(paths: list[Path], masks: np.ndarray, frames: list[int], output_path: Path) -> None:
    cells: list[np.ndarray] = []
    valid_frames = [idx for idx in frames if 0 <= idx < len(paths)]
    if not valid_frames:
        valid_frames = np.linspace(0, len(paths) - 1, min(6, len(paths)), dtype=int).tolist()
    for idx in valid_frames:
        image_bgr = cv2.imread(str(paths[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = masks[idx].astype(bool)
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray([255, 0, 0])).astype(np.uint8)
        cv2.putText(
            overlay,
            f"f{idx} area={mask.mean():.3f}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(overlay)
    montage = np.concatenate(cells, axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    frame_dir = args.frame_dir.resolve() if args.frame_dir is not None else processed_demo_dir / "original_images"
    paths = frame_paths(frame_dir)
    if not (0 <= args.prompt_frame < len(paths)):
        raise ValueError(f"--prompt-frame must be in [0, {len(paths) - 1}], got {args.prompt_frame}")

    segmentation_dir = processed_demo_dir / "segmentation_processor"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    masks_by_frame = run_sam31(
        frame_dir,
        args.checkpoint.resolve(),
        args.prompt,
        args.prompt_frame,
        args.max_num_objects,
    )
    masks = make_union_masks(paths, masks_by_frame, args.close_kernel, args.post_dilation)

    output_path = segmentation_dir / args.output_name
    np.save(output_path, masks)
    montage_path = segmentation_dir / f"{Path(args.output_name).stem}_montage.jpg"
    write_montage(paths, masks, args.montage_frames, montage_path)

    if args.replace:
        target = segmentation_dir / "masks_arm.npy"
        if target.exists():
            backup = segmentation_dir / "masks_arm.before_sam31.npy"
            if not backup.exists():
                shutil.copy2(target, backup)
        shutil.copy2(output_path, target)

    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "frame_dir": str(frame_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt": args.prompt,
        "prompt_frame": int(args.prompt_frame),
        "frames": int(len(paths)),
        "output": str(output_path),
        "montage": str(montage_path),
        "mean_area": float(masks.mean()),
        "min_area": float(masks.mean(axis=(1, 2)).min()),
        "max_area": float(masks.mean(axis=(1, 2)).max()),
        "objects_per_frame": {str(idx): len(masks_by_frame.get(idx, {})) for idx in range(len(paths))},
        "replace": bool(args.replace),
    }
    summary_path = segmentation_dir / f"{Path(args.output_name).stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
