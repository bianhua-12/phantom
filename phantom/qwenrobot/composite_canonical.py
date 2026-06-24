from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INVALID_MARKER = "INVALID_NO_HDF5.json"
KEY_FRAMES = (27, 28, 29, 30, 31, 52, 75, 98)


class CompositeInputError(RuntimeError):
    pass


def require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python (cv2) is required for video/depth image compositing.") from exc
    return cv2


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def save_video(frames: list[np.ndarray], output_video: Path, fps: float) -> None:
    import imageio.v2 as imageio

    output_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_video), frames, fps=float(fps), quality=8, macro_block_size=1)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def read_video_frames(path: Path) -> list[np.ndarray]:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def overlay_robot(raw: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[robot_mask.astype(bool)] = robot_rgb[robot_mask.astype(bool)]
    return out


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[np.asarray(mask).astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def assert_not_invalid_artifact(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / INVALID_MARKER).exists():
            raise CompositeInputError(f"Refusing invalid no-HDF5 artifact under {parent}")
    if "invalid_no_hdf5" in str(resolved).lower():
        raise CompositeInputError(f"Refusing invalid no-HDF5 artifact path: {resolved}")


def load_canonical_cache(path: Path) -> dict[str, np.ndarray | str]:
    assert_not_invalid_artifact(path)
    if not path.exists():
        raise CompositeInputError(f"canonical npz does not exist: {path}")
    arrays = np.load(path, allow_pickle=False)
    required = ["robot_rgb", "robot_mask", "robot_depth", "qpos", "frame_indices"]
    missing = [key for key in required if key not in arrays.files]
    if missing:
        raise CompositeInputError(f"canonical npz missing required arrays: {missing}")
    robot_rgb = arrays["robot_rgb"].astype(np.uint8)
    robot_mask = arrays["robot_mask"].astype(bool)
    robot_depth = arrays["robot_depth"].astype(np.float32)
    qpos = arrays["qpos"].astype(np.float32)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    if robot_rgb.ndim != 4 or robot_rgb.shape[-1] != 3:
        raise CompositeInputError(f"robot_rgb must have shape (T,H,W,3), got {robot_rgb.shape}")
    if robot_mask.shape != robot_rgb.shape[:3]:
        raise CompositeInputError(f"robot_mask shape {robot_mask.shape} does not match robot_rgb {robot_rgb.shape}")
    if robot_depth.shape != robot_mask.shape:
        raise CompositeInputError(f"robot_depth shape {robot_depth.shape} does not match robot_mask {robot_mask.shape}")
    if len(qpos) != len(robot_rgb) or len(frame_indices) != len(robot_rgb):
        raise CompositeInputError("qpos, frame_indices, and robot_rgb must have the same frame count")
    return {
        "robot_rgb": robot_rgb,
        "robot_mask": robot_mask,
        "robot_depth": robot_depth,
        "qpos": qpos,
        "frame_indices": frame_indices,
        "cache_hash": str(arrays["cache_hash"].item()) if "cache_hash" in arrays.files and arrays["cache_hash"].shape == () else "",
    }


def load_scene_depth(path: Path, n_frames: int, height: int, width: int) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise CompositeInputError(f"scene depth must have shape (T,H,W), got {depth.shape}: {path}")
    if len(depth) < n_frames:
        raise CompositeInputError(f"scene depth has {len(depth)} frames, expected at least {n_frames}: {path}")
    depth = depth[:n_frames]
    if depth.shape[1:3] != (height, width):
        cv2 = require_cv2()
        depth = np.asarray([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth], dtype=np.float32)
    if not np.isfinite(depth).all():
        raise CompositeInputError(f"scene depth contains non-finite values: {path}")
    return depth


def compose_depth_overlay(
    background: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    scene_depth: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene_depth.shape != robot_depth.shape:
        cv2 = require_cv2()
        scene_depth = cv2.resize(scene_depth.astype(np.float32), robot_depth.shape[::-1], interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(robot_depth) & np.isfinite(scene_depth) & (robot_depth > 0.0)
    visible = robot_mask.astype(bool) & valid & (robot_depth <= scene_depth + float(margin))
    occluded = robot_mask.astype(bool) & ~visible
    overlay = background.copy()
    overlay[visible] = robot_rgb[visible]
    return overlay, visible.astype(bool), occluded.astype(bool)


def make_montage(
    output: Path,
    frames: list[int],
    background: list[np.ndarray],
    robot_rgb: np.ndarray,
    nodepth: list[np.ndarray],
    scene_depth: np.ndarray,
    visible: np.ndarray,
    depth_overlay: list[np.ndarray],
) -> None:
    cv2 = require_cv2()
    rows = []
    for frame_i in frames:
        cells = [
            background[frame_i],
            robot_rgb[frame_i],
            nodepth[frame_i],
            colorize_depth(scene_depth[frame_i]),
            mask_rgb(visible[frame_i], (0, 255, 0)),
            depth_overlay[frame_i],
        ]
        labels = ["clean", "canonical robot", "no-depth", "scene depth", "visible mask", "depth"]
        labelled = []
        for label, img in zip(labels, cells):
            cell = img.copy()
            cv2.putText(cell, f"{label} f{frame_i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cell, f"{label} f{frame_i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            labelled.append(cell)
        rows.append(np.concatenate(labelled, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite no-depth and depth overlays from a canonical robot cache.")
    parser.add_argument("--canonical-npz", type=Path, required=True)
    parser.add_argument("--background-video", type=Path, required=True)
    parser.add_argument("--scene-depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-occlusion-margin", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_npz = args.canonical_npz.resolve()
    output_dir = args.output_dir.resolve()
    cache = load_canonical_cache(canonical_npz)
    robot_rgb = cache["robot_rgb"]  # type: ignore[assignment]
    robot_mask = cache["robot_mask"]  # type: ignore[assignment]
    robot_depth = cache["robot_depth"]  # type: ignore[assignment]
    qpos = cache["qpos"]  # type: ignore[assignment]
    frame_indices = cache["frame_indices"]  # type: ignore[assignment]

    background = read_video_frames(args.background_video.resolve())
    n_frames = min(len(background), len(robot_rgb))
    if n_frames <= 0:
        raise CompositeInputError("No frames available for compositing")
    background = background[:n_frames]
    robot_rgb = robot_rgb[:n_frames]
    robot_mask = robot_mask[:n_frames]
    robot_depth = robot_depth[:n_frames]
    qpos = qpos[:n_frames]
    frame_indices = frame_indices[:n_frames]
    height, width = background[0].shape[:2]
    if robot_rgb.shape[1:3] != (height, width):
        raise CompositeInputError(
            f"background resolution {(height, width)} does not match canonical robot {robot_rgb.shape[1:3]}"
        )

    scene_depth = load_scene_depth(args.scene_depth.resolve(), n_frames, height, width)
    nodepth_overlay = [overlay_robot(background[i], robot_rgb[i], robot_mask[i]) for i in range(n_frames)]
    depth_overlay: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []
    occlusion_masks: list[np.ndarray] = []
    for frame_i in range(n_frames):
        overlay, visible, occluded = compose_depth_overlay(
            background[frame_i],
            robot_rgb[frame_i],
            robot_mask[frame_i],
            robot_depth[frame_i],
            scene_depth[frame_i],
            args.depth_occlusion_margin,
        )
        depth_overlay.append(overlay)
        visible_masks.append(visible)
        occlusion_masks.append(occluded)

    visible_arr = np.asarray(visible_masks, dtype=bool)
    occlusion_arr = np.asarray(occlusion_masks, dtype=bool)
    canonical_mask_hash = array_hash(robot_mask, qpos)
    composite_npz = output_dir / "canonical_composite_masks.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        composite_npz,
        visible_mask=visible_arr,
        occlusion_mask=occlusion_arr,
        robot_mask=robot_mask.astype(bool),
        frame_indices=frame_indices.astype(np.int64),
        canonical_mask_qpos_hash=canonical_mask_hash,
        source_cache_hash=str(cache.get("cache_hash", "")),
    )

    nodepth_video = output_dir / "video_overlay_Kinova3_shoulders_canonical_nodepth.mp4"
    depth_video = output_dir / "video_overlay_Kinova3_shoulders_canonical_depth.mp4"
    save_video(nodepth_overlay, nodepth_video, args.fps)
    save_video(depth_overlay, depth_video, args.fps)
    montage_frames = [i for i in KEY_FRAMES if i < n_frames]
    if not montage_frames:
        montage_frames = [0, n_frames // 2, n_frames - 1]
        montage_frames = sorted(set(montage_frames))
    montage = output_dir / "canonical_composite_montage.jpg"
    make_montage(montage, montage_frames, background, robot_rgb, nodepth_overlay, scene_depth, visible_arr, depth_overlay)

    summary: dict[str, Any] = {
        "stage": "qwenrobot_composite_canonical",
        "canonical_npz": str(canonical_npz),
        "background_video": str(args.background_video.resolve()),
        "scene_depth": str(args.scene_depth.resolve()),
        "output_dir": str(output_dir),
        "frames": int(n_frames),
        "fps": float(args.fps),
        "resolution": [int(width), int(height)],
        "depth_occlusion_margin": float(args.depth_occlusion_margin),
        "depth_formula": "visible = robot_mask & finite(D_robot) & finite(D_scene) & (D_robot <= D_scene + margin)",
        "mean_robot_mask_area": float(robot_mask.mean()),
        "mean_visible_mask_area": float(visible_arr.mean()),
        "mean_occlusion_mask_area": float(occlusion_arr.mean()),
        "canonical_mask_qpos_hash": canonical_mask_hash,
        "source_cache_hash": str(cache.get("cache_hash", "")),
        "composite_npz": str(composite_npz),
        "nodepth_overlay": str(nodepth_video),
        "depth_overlay": str(depth_video),
        "montage": str(montage),
        "invalid_artifact_policy": "Inputs with INVALID_NO_HDF5.json or invalid_no_hdf5 in their path are rejected.",
        "visual_alignment_note": "SAM/ProPainter inputs, when used upstream, are SAM3-family approximate implementation artifacts.",
    }
    summary_path = output_dir / "canonical_composite_summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
