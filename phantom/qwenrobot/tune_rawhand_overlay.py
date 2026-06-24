from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy as media
import numpy as np


def configure_mediapy_ffmpeg() -> None:
    current_ffmpeg = getattr(media._config, "ffmpeg_name_or_path", None)
    if current_ffmpeg and current_ffmpeg != "ffmpeg":
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "tune_rawhand_overlay requires imageio-ffmpeg when mediapy does "
            "not have an explicit ffmpeg binary configured."
        ) from exc

    media._config.ffmpeg_name_or_path = imageio_ffmpeg.get_ffmpeg_exe()


DEFAULT_FRAMES = [0, 23, 47, 71, 95, 119]
DEFAULT_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_OFFSETS = [0.0, 0.1, 0.2, 0.3]
DEFAULT_MARGINS = [0.03, 0.08, 0.15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune raw-hand Phantom depth overlay from saved debug arrays.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("outputs/phantom_egodex_exact_epic/processed/egodex_phantom/0"),
    )
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=DEFAULT_FRAMES)
    parser.add_argument("--scales", type=float, nargs="*", default=DEFAULT_SCALES)
    parser.add_argument("--offsets", type=float, nargs="*", default=DEFAULT_OFFSETS)
    parser.add_argument("--margins", type=float, nargs="*", default=DEFAULT_MARGINS)
    parser.add_argument("--target-visible-min", type=float, default=0.65)
    parser.add_argument("--target-visible-max", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def load_debug_arrays(debug_dir: Path) -> dict[str, np.ndarray]:
    required = {
        "robot_mask": np.uint8,
        "scene_depth": np.float32,
        "robot_depth": np.float32,
        "robot_rgb": np.uint8,
    }
    arrays = {}
    for name, dtype in required.items():
        path = debug_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[name] = np.asarray(np.load(path), dtype=dtype)
    return arrays


def load_raw_frames(demo_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    video_path = demo_dir / "video_rgb_imgs.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    frames = np.asarray(media.read_video(str(video_path)), dtype=np.uint8)
    h, w = shape
    if frames.shape[1:3] != (h, w):
        frames = np.asarray([cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR) for frame in frames], dtype=np.uint8)
    return frames


def make_visible_mask(
    robot_mask: np.ndarray,
    scene_depth: np.ndarray,
    robot_depth: np.ndarray,
    scale: float,
    offset: float,
    margin: float,
) -> np.ndarray:
    scene_calibrated = scene_depth * float(scale) + float(offset)
    valid = (
        (robot_mask > 0)
        & np.isfinite(scene_calibrated)
        & np.isfinite(robot_depth)
        & (scene_calibrated > 0.0)
        & (robot_depth > 0.0)
    )
    return valid & (robot_depth <= scene_calibrated + float(margin))


def compose_overlay(raw: np.ndarray, robot_rgb: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    overlay = raw.copy()
    overlay[visible_mask] = robot_rgb[visible_mask]
    return overlay


def evaluate_params(
    robot_mask: np.ndarray,
    scene_depth: np.ndarray,
    robot_depth: np.ndarray,
    scale: float,
    offset: float,
    margin: float,
    target_min: float,
    target_max: float,
) -> dict[str, float]:
    visible = make_visible_mask(robot_mask, scene_depth, robot_depth, scale, offset, margin)
    robot_pixels = robot_mask.reshape(robot_mask.shape[0], -1).sum(axis=1).astype(np.float64)
    visible_pixels = visible.reshape(visible.shape[0], -1).sum(axis=1).astype(np.float64)
    visible_fraction = visible_pixels / np.maximum(robot_pixels, 1.0)
    target_mid = 0.5 * (target_min + target_max)
    below = np.maximum(0.0, target_min - visible_fraction)
    above = np.maximum(0.0, visible_fraction - target_max)
    score = (
        abs(float(visible_fraction.mean()) - target_mid)
        + 0.75 * float(below.mean() + above.mean())
        + 0.20 * float(visible_fraction.std())
    )
    return {
        "scene_depth_scale": float(scale),
        "scene_depth_offset": float(offset),
        "depth_occlusion_margin": float(margin),
        "score": float(score),
        "visible_fraction_mean": float(visible_fraction.mean()),
        "visible_fraction_min": float(visible_fraction.min()),
        "visible_fraction_max": float(visible_fraction.max()),
        "visible_area_mean": float(visible.mean()),
        "robot_area_mean": float((robot_mask > 0).mean()),
    }


def label_cell(cell: np.ndarray, text: str) -> np.ndarray:
    out = cell.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(path), frames, fps=fps, codec="ffv1")


def write_geometry_outputs(
    demo_dir: Path,
    output_dir: Path,
    raw_frames: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    frame_ids: list[int],
    fps: int,
    skip_videos: bool,
) -> np.ndarray:
    nodepth_frames = np.asarray(
        [compose_overlay(raw_frames[i], robot_rgb[i], robot_mask[i] > 0) for i in range(len(raw_frames))],
        dtype=np.uint8,
    )
    if not skip_videos:
        write_video(demo_dir / "video_overlay_Kinova3_shoulders_rawhand_nodepth.mkv", nodepth_frames, fps)

    rows = []
    for label, frames in (
        ("raw", raw_frames),
        ("robot rgb", robot_rgb),
        ("nodepth", nodepth_frames),
        ("robot mask", np.asarray([mask_to_rgb(mask > 0) for mask in robot_mask], dtype=np.uint8)),
    ):
        cells = [label_cell(frames[idx], f"{label} f{idx}") for idx in frame_ids]
        rows.append(np.concatenate(cells, axis=1))
    media.write_image(str(output_dir / "geometry_montage.jpg"), np.concatenate(rows, axis=0))
    return nodepth_frames


def write_depth_outputs(
    demo_dir: Path,
    output_dir: Path,
    raw_frames: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    scene_depth: np.ndarray,
    robot_depth: np.ndarray,
    top_rows: list[dict[str, float]],
    frame_ids: list[int],
    fps: int,
    skip_videos: bool,
) -> None:
    best = top_rows[0]
    best_mask = make_visible_mask(
        robot_mask,
        scene_depth,
        robot_depth,
        best["scene_depth_scale"],
        best["scene_depth_offset"],
        best["depth_occlusion_margin"],
    )
    best_frames = np.asarray(
        [compose_overlay(raw_frames[i], robot_rgb[i], best_mask[i]) for i in range(len(raw_frames))],
        dtype=np.uint8,
    )
    if not skip_videos:
        write_video(demo_dir / "video_overlay_Kinova3_shoulders_rawhand_depth_tuned.mkv", best_frames, fps)

    rows = []
    for rank, row in enumerate(top_rows, start=1):
        visible = make_visible_mask(
            robot_mask,
            scene_depth,
            robot_depth,
            row["scene_depth_scale"],
            row["scene_depth_offset"],
            row["depth_occlusion_margin"],
        )
        overlays = np.asarray(
            [compose_overlay(raw_frames[i], robot_rgb[i], visible[i]) for i in range(len(raw_frames))],
            dtype=np.uint8,
        )
        label = (
            f"top{rank} s={row['scene_depth_scale']:.2g} "
            f"o={row['scene_depth_offset']:.2g} m={row['depth_occlusion_margin']:.2g} "
            f"vf={row['visible_fraction_mean']:.2f}"
        )
        rows.append(np.concatenate([label_cell(overlays[idx], f"{label} f{idx}") for idx in frame_ids], axis=1))
        rows.append(np.concatenate([label_cell(mask_to_rgb(visible[idx]), f"visible top{rank} f{idx}") for idx in frame_ids], axis=1))
    media.write_image(str(output_dir / "depth_sweep_topk.jpg"), np.concatenate(rows, axis=0))


def main() -> None:
    configure_mediapy_ffmpeg()
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    debug_dir = args.debug_dir.resolve() if args.debug_dir else demo_dir / "inpaint_processor" / "depth_overlay_debug_rawhand"
    output_dir = args.output_dir.resolve() if args.output_dir else demo_dir / "inpaint_processor" / "rawhand_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_debug_arrays(debug_dir)
    robot_mask = arrays["robot_mask"] > 0
    scene_depth = arrays["scene_depth"].astype(np.float32)
    robot_depth = arrays["robot_depth"].astype(np.float32)
    robot_rgb = arrays["robot_rgb"].astype(np.uint8)
    raw_frames = load_raw_frames(demo_dir, robot_mask.shape[1:3])
    n_frames = min(len(raw_frames), len(robot_rgb), len(robot_mask), len(scene_depth), len(robot_depth))
    raw_frames = raw_frames[:n_frames]
    robot_rgb = robot_rgb[:n_frames]
    robot_mask = robot_mask[:n_frames]
    scene_depth = scene_depth[:n_frames]
    robot_depth = robot_depth[:n_frames]
    frame_ids = [idx for idx in args.frames if 0 <= idx < n_frames]
    if not frame_ids:
        frame_ids = np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int).tolist()

    write_geometry_outputs(
        demo_dir,
        output_dir,
        raw_frames,
        robot_rgb,
        robot_mask,
        frame_ids,
        args.fps,
        args.skip_videos,
    )

    rows = []
    for scale in args.scales:
        for offset in args.offsets:
            for margin in args.margins:
                rows.append(
                    evaluate_params(
                        robot_mask,
                        scene_depth,
                        robot_depth,
                        scale,
                        offset,
                        margin,
                        args.target_visible_min,
                        args.target_visible_max,
                    )
                )
    rows = sorted(rows, key=lambda row: row["score"])
    top_rows = rows[: max(1, args.top_k)]
    write_depth_outputs(
        demo_dir,
        output_dir,
        raw_frames,
        robot_rgb,
        robot_mask,
        scene_depth,
        robot_depth,
        top_rows,
        frame_ids,
        args.fps,
        args.skip_videos,
    )

    report = {
        "demo_dir": str(demo_dir),
        "debug_dir": str(debug_dir),
        "frames": int(n_frames),
        "selected_frames": frame_ids,
        "target_visible_fraction": [float(args.target_visible_min), float(args.target_visible_max)],
        "best": top_rows[0],
        "top": top_rows,
        "all": rows,
        "outputs": {
            "nodepth_video": str(demo_dir / "video_overlay_Kinova3_shoulders_rawhand_nodepth.mkv"),
            "depth_tuned_video": str(demo_dir / "video_overlay_Kinova3_shoulders_rawhand_depth_tuned.mkv"),
            "geometry_montage": str(output_dir / "geometry_montage.jpg"),
            "depth_sweep_topk": str(output_dir / "depth_sweep_topk.jpg"),
        },
    }
    (output_dir / "tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": top_rows[0], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
