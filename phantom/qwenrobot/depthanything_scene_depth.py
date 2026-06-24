from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_DEPTHANYTHING_ROOT = Path("/mnt/project_rlinf/jlchen/code/WAFT/thirdparty/DepthAnythingV2")


def frame_paths(frame_dir: Path) -> list[Path]:
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        paths = sorted(frame_dir.glob("*.png"), key=lambda path: int(path.stem))
    return paths


def read_frames(processed_demo_dir: Path) -> list[np.ndarray]:
    image_dir = processed_demo_dir / "original_images"
    paths = frame_paths(image_dir)
    if paths:
        frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {path}")
            frames.append(frame)
        return frames

    video_path = processed_demo_dir / "video_rgb_imgs.mkv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No original_images and cannot open {video_path}")
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
        raise RuntimeError(f"No frames read from {video_path}")
    return frames


def load_hand_samples(processed_demo_dir: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    hand_dir = processed_demo_dir / "hand_processor"
    for side in ("left", "right"):
        path = hand_dir / f"hand_data_{side}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        detected = data["hand_detected"].astype(bool)
        kpts_2d = data["kpts_2d"].astype(np.float32)
        kpts_3d = data["kpts_3d"].astype(np.float32)
        for frame_idx in np.where(detected)[0]:
            points = kpts_2d[frame_idx]
            depths = kpts_3d[frame_idx, :, 2]
            finite = np.isfinite(points).all(axis=1) & np.isfinite(depths)
            finite &= depths > 0.03
            finite &= points[:, 0] >= 0
            finite &= points[:, 0] < width
            finite &= points[:, 1] >= 0
            finite &= points[:, 1] < height
            if finite.any():
                xs.append(points[finite, 0])
                ys.append(depths[finite])
                frame_ids.append(np.full(int(finite.sum()), int(frame_idx), dtype=np.int64))
    if not xs:
        raise RuntimeError(f"No valid hand 2D/3D depth samples in {processed_demo_dir}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(frame_ids)


def sample_relative_depth(relative: np.ndarray, processed_demo_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    height, width = relative.shape[1:]
    sampled: list[np.ndarray] = []
    target: list[np.ndarray] = []
    hand_dir = processed_demo_dir / "hand_processor"
    for side in ("left", "right"):
        path = hand_dir / f"hand_data_{side}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        detected = data["hand_detected"].astype(bool)
        kpts_2d = data["kpts_2d"].astype(np.float32)
        kpts_3d = data["kpts_3d"].astype(np.float32)
        for frame_idx in np.where(detected)[0]:
            if frame_idx >= len(relative):
                continue
            points = kpts_2d[frame_idx]
            depths = kpts_3d[frame_idx, :, 2]
            finite = np.isfinite(points).all(axis=1) & np.isfinite(depths)
            finite &= depths > 0.03
            finite &= points[:, 0] >= 0
            finite &= points[:, 0] < width
            finite &= points[:, 1] >= 0
            finite &= points[:, 1] < height
            if not finite.any():
                continue
            px = np.clip(np.rint(points[finite, 0]).astype(np.int64), 0, width - 1)
            py = np.clip(np.rint(points[finite, 1]).astype(np.int64), 0, height - 1)
            sampled.append(relative[frame_idx, py, px])
            target.append(depths[finite])
    if not sampled:
        raise RuntimeError("No valid samples for depth calibration")
    return np.concatenate(sampled).astype(np.float64), np.concatenate(target).astype(np.float64)


def robust_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 30:
        raise RuntimeError(f"Need at least 30 depth calibration samples, got {len(x)}")
    keep = np.ones(len(x), dtype=bool)
    coef = np.asarray([1.0, 0.0], dtype=np.float64)
    for _ in range(4):
        coef = np.polyfit(x[keep], y[keep], deg=1)
        pred = coef[0] * x + coef[1]
        resid = np.abs(pred - y)
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med))) + 1e-6
        keep = resid <= med + 3.0 * 1.4826 * mad
        if keep.sum() < 30:
            keep[:] = True
            break
    pred = coef[0] * x + coef[1]
    mae = float(np.mean(np.abs(pred[keep] - y[keep])))
    return float(coef[0]), float(coef[1]), mae


def calibrate_metric_depth(relative: np.ndarray, processed_demo_dir: Path) -> tuple[np.ndarray, dict[str, float]]:
    rel_samples, metric_samples = sample_relative_depth(relative, processed_demo_dir)
    candidates = []
    for polarity, values in (("direct", rel_samples), ("inverse_sign", -rel_samples)):
        scale, bias, mae = robust_linear_fit(values, metric_samples)
        candidates.append((mae, polarity, scale, bias))
    mae, polarity, scale, bias = min(candidates, key=lambda row: row[0])
    source = relative if polarity == "direct" else -relative
    metric = source.astype(np.float32) * np.float32(scale) + np.float32(bias)
    metric = np.clip(metric, 0.15, 10.0).astype(np.float32)
    if not np.isfinite(metric).all():
        raise RuntimeError("Metric depth contains non-finite values")
    if float(metric.std()) < 1e-4:
        raise RuntimeError("Metric depth is effectively constant after calibration")
    return metric, {
        "calibration_polarity": polarity,
        "calibration_scale": float(scale),
        "calibration_bias": float(bias),
        "calibration_mae_m": float(mae),
        "calibration_samples": int(len(rel_samples)),
    }


def load_depth_model(root: Path, checkpoint: Path, encoder: str):
    if not root.exists():
        raise FileNotFoundError(root)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(root))
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if encoder not in configs:
        raise ValueError(f"Unsupported encoder {encoder!r}; choose one of {sorted(configs)}")
    model = DepthAnythingV2(**configs[encoder])
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model


def write_montage(frames: list[np.ndarray], metric: np.ndarray, output_path: Path) -> None:
    ids = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int).tolist()
    cells = []
    for idx in ids:
        rgb = frames[idx]
        depth = metric[idx]
        lo, hi = np.percentile(depth[np.isfinite(depth)], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cell = np.concatenate([rgb, color], axis=0)
        cv2.putText(cell, f"f{idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(cells, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scene depth for a processed EgoDex demo with DepthAnythingV2.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--depthanything-root", type=Path, default=DEFAULT_DEPTHANYTHING_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--output-name", type=str, default="depth.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    frames = read_frames(processed_demo_dir)
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError("All frames must have the same dimensions")

    model = load_depth_model(args.depthanything_root.resolve(), args.checkpoint.resolve(), args.encoder)
    relative = []
    with torch.inference_mode():
        for frame in frames:
            relative.append(model.infer_image(frame, input_size=args.input_size).astype(np.float32))
    relative_depth = np.stack(relative).astype(np.float32)
    metric_depth, calibration = calibrate_metric_depth(relative_depth, processed_demo_dir)

    depth_dir = processed_demo_dir / "depth_processor"
    depth_dir.mkdir(parents=True, exist_ok=True)
    relative_path = depth_dir / "scene_depth_relative.npy"
    metric_path = depth_dir / "scene_depth_metric.npy"
    output_path = processed_demo_dir / args.output_name
    np.save(relative_path, relative_depth)
    np.save(metric_path, metric_depth)
    np.save(output_path, metric_depth)
    montage_path = depth_dir / "scene_depth_montage.jpg"
    write_montage(frames, metric_depth, montage_path)

    finite = np.isfinite(metric_depth)
    summary = {
        "processed_demo_dir": str(processed_demo_dir),
        "depthanything_root": str(args.depthanything_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "encoder": args.encoder,
        "input_size": int(args.input_size),
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
        **calibration,
    }
    summary_path = depth_dir / "scene_depth_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
