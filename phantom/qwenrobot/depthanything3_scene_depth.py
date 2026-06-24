from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from phantom.qwenrobot.depthanything_scene_depth import (
    calibrate_metric_depth,
    frame_paths,
    read_frames,
    write_montage,
)


DEFAULT_DA3_MODEL_ID = "depth-anything/DA3METRIC-LARGE"


def read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from input video: {path}")
    return frames


def image_inputs(
    processed_demo_dir: Path,
    input_video: Path | None,
) -> tuple[list[str] | list[np.ndarray], list[np.ndarray], str]:
    if input_video is not None:
        frames_bgr = read_video_frames(input_video)
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        return frames_rgb, frames_bgr, str(input_video)

    image_dir = processed_demo_dir / "original_images"
    paths = frame_paths(image_dir)
    if paths:
        frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {path}")
            frames.append(frame)
        return [str(path) for path in paths], frames, str(image_dir)

    frames_bgr = read_frames(processed_demo_dir)
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    return frames_rgb, frames_bgr, str(processed_demo_dir / "video_rgb_imgs.mkv")


def add_depthanything3_to_path(root: Path | None) -> None:
    if root is None:
        return
    root = root.resolve()
    candidates = [root / "src", root]
    for candidate in candidates:
        if (candidate / "depth_anything_3").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(f"Cannot find depth_anything_3 package under {root}")


def load_da3_model(model_id: str, root: Path | None):
    add_depthanything3_to_path(root)
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import depth_anything_3. Install Depth Anything 3 or pass "
            "--depthanything3-root pointing at the cloned repo."
        ) from exc

    model = DepthAnything3.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device=device).eval()


def read_focal_length_px(path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "left" in payload:
        item = payload["left"]
    else:
        item = payload
    fx = float(item["fx"])
    fy = float(item["fy"])
    return 0.5 * (fx + fy)


def resize_depth_stack(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape (T,H,W), got {depth.shape}")
    if depth.shape[1:] == (height, width):
        return depth.astype(np.float32)
    resized = [
        cv2.resize(frame.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        for frame in depth
    ]
    return np.stack(resized).astype(np.float32)


def infer_depth(
    model,
    inputs: list[str] | list[np.ndarray],
    height: int,
    width: int,
    *,
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
    chunk_size: int,
) -> np.ndarray:
    depths = []
    total = len(inputs)
    if chunk_size <= 0:
        chunk_size = total
    with torch.inference_mode():
        for start in range(0, total, chunk_size):
            chunk = inputs[start : start + chunk_size]
            prediction = model.inference(
                chunk,
                process_res=process_res,
                process_res_method=process_res_method,
                ref_view_strategy=ref_view_strategy,
                export_format="mini_npz",
            )
            depths.append(resize_depth_stack(np.asarray(prediction.depth, dtype=np.float32), height, width))
    return np.concatenate(depths, axis=0).astype(np.float32)


def metric_from_raw_depth(
    raw_depth: np.ndarray,
    processed_demo_dir: Path,
    mode: str,
    focal_length_px: float | None,
) -> tuple[np.ndarray, dict[str, float | str | int | None]]:
    if mode == "hand-calibrated":
        metric_depth, calibration = calibrate_metric_depth(raw_depth, processed_demo_dir)
        return metric_depth, {"metric_mode": mode, **calibration}

    if mode == "focal-scaled":
        if focal_length_px is None:
            raise ValueError("--metric-mode focal-scaled requires --focal-length-px or --camera-intrinsics")
        metric_depth = raw_depth.astype(np.float32) * np.float32(focal_length_px)
    elif mode == "direct":
        metric_depth = raw_depth.astype(np.float32)
    else:
        raise ValueError(f"Unsupported metric mode: {mode}")

    metric_depth = np.clip(metric_depth, 0.15, 10.0).astype(np.float32)
    if not np.isfinite(metric_depth).all():
        raise RuntimeError("Metric depth contains non-finite values")
    if float(metric_depth.std()) < 1e-4:
        raise RuntimeError("Metric depth is effectively constant")
    return metric_depth, {
        "metric_mode": mode,
        "focal_length_px": float(focal_length_px) if focal_length_px is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scene depth for a processed EgoDex demo with Depth Anything 3.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--depthanything3-root", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_DA3_MODEL_ID)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", type=str, default="middle")
    parser.add_argument("--chunk-size", type=int, default=16, help="Use 0 to run all frames in one DA3 inference call.")
    parser.add_argument(
        "--metric-mode",
        choices=("hand-calibrated", "direct", "focal-scaled"),
        default="hand-calibrated",
        help="hand-calibrated preserves the existing 2D/3D keypoint metric calibration guardrail.",
    )
    parser.add_argument("--camera-intrinsics", type=Path, default=None)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--output-name", type=str, default="depth.npy")
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Optional video to use as scene-depth input instead of original_images/video_rgb_imgs.mkv.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Optional prefix for depth_processor artifacts, e.g. qwen_da3_clean_direct.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
    processed_demo_dir = args.processed_demo_dir.resolve()
    input_video = args.input_video.resolve() if args.input_video is not None else None
    inputs, frames, input_source = image_inputs(processed_demo_dir, input_video)
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError("All frames must have the same dimensions")

    model = load_da3_model(args.model_id, args.depthanything3_root)
    raw_depth = infer_depth(
        model,
        inputs,
        height,
        width,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
        chunk_size=args.chunk_size,
    )
    focal_length_px = args.focal_length_px
    if focal_length_px is None:
        focal_length_px = read_focal_length_px(args.camera_intrinsics.resolve()) if args.camera_intrinsics else None
    metric_depth, metric_info = metric_from_raw_depth(
        raw_depth,
        processed_demo_dir,
        args.metric_mode,
        focal_length_px,
    )

    depth_dir = processed_demo_dir / "depth_processor"
    depth_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.artifact_prefix}_" if args.artifact_prefix else ""
    relative_path = depth_dir / f"{prefix}scene_depth_relative.npy"
    metric_path = depth_dir / f"{prefix}scene_depth_metric.npy"
    output_path = processed_demo_dir / args.output_name
    np.save(relative_path, raw_depth.astype(np.float32))
    np.save(metric_path, metric_depth)
    np.save(output_path, metric_depth)
    montage_path = depth_dir / f"{prefix}scene_depth_montage.jpg"
    write_montage(frames, metric_depth, montage_path)

    finite = np.isfinite(metric_depth)
    summary = {
        "backend": "depthanything-v3",
        "processed_demo_dir": str(processed_demo_dir),
        "input_source": input_source,
        "input_video": str(input_video) if input_video is not None else None,
        "artifact_prefix": args.artifact_prefix,
        "depthanything3_root": str(args.depthanything3_root.resolve()) if args.depthanything3_root else None,
        "model_id": args.model_id,
        "hf_home": str(args.hf_home.resolve()) if args.hf_home else os.environ.get("HF_HOME"),
        "process_res": int(args.process_res),
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
        "chunk_size": int(args.chunk_size),
        "frames": int(len(frames)),
        "shape": list(metric_depth.shape),
        "finite_ratio": float(finite.mean()),
        "min_depth_m": float(metric_depth[finite].min()),
        "median_depth_m": float(np.median(metric_depth[finite])),
        "max_depth_m": float(metric_depth[finite].max()),
        "std_depth_m": float(metric_depth[finite].std()),
        "relative_output": str(relative_path),
        "metric_output": str(metric_path),
        "paths_depth_output": str(output_path),
        "montage": str(montage_path),
        **metric_info,
    }
    summary_path = depth_dir / f"{prefix}scene_depth_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
