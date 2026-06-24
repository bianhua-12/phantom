from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mediapy as media
import numpy as np

from phantom.qwenrobot.prepare_egodex_for_phantom import DEFAULT_EGODEX_ROOT, REPO_ROOT, discover_demos


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "phantom_egodex_exact_epic_demo10"


def parse_indices(value: str, n_demos: int) -> list[int]:
    if value == "all":
        return list(range(n_demos))
    out: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    bad = [idx for idx in out if idx < 0 or idx >= n_demos]
    if bad:
        raise ValueError(f"Demo indices out of range 0..{n_demos - 1}: {bad}")
    return sorted(dict.fromkeys(out))


def run(cmd: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def export_mp4(mkv_path: Path, mp4_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(mkv_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(mp4_path),
        ]
    )


def write_filled_preview(root: Path, fps: float) -> Path:
    mkv_path = root / "video_overlay_Kinova3_shoulders.mkv"
    training_path = root / "inpaint_processor" / "training_data_shoulders.npz"
    frames = media.read_video(mkv_path)
    valid = np.load(training_path, allow_pickle=True)["valid"].astype(bool)
    if len(frames) != len(valid):
        raise RuntimeError(f"Frame count mismatch: video={len(frames)} valid={len(valid)}")

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise RuntimeError(f"No valid frames in {training_path}")

    filled = frames.copy()
    for idx in np.where(~valid)[0]:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
        filled[idx] = frames[nearest]

    tmp_path = root / "video_overlay_Kinova3_shoulders_preview_filled_tmp.mp4"
    final_path = root / "video_overlay_Kinova3_shoulders_preview_filled.mp4"
    media.write_video(tmp_path, filled, fps=fps, codec="libx264", crf=18)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(final_path),
        ]
    )
    return final_path


def summarize_demo(processed_demo_dir: Path) -> dict[str, object]:
    training_path = processed_demo_dir / "inpaint_processor" / "training_data_shoulders.npz"
    valid = np.load(training_path, allow_pickle=True)["valid"].astype(bool)
    invalid = np.where(~valid)[0].astype(int).tolist()
    rawhand_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4"
    return {
        "processed_demo_dir": str(processed_demo_dir),
        "overlay_mkv": str(processed_demo_dir / "video_overlay_Kinova3_shoulders.mkv"),
        "overlay_mp4": str(processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4"),
        "preview_filled_mp4": str(processed_demo_dir / "video_overlay_Kinova3_shoulders_preview_filled.mp4"),
        "rawhand_overlay_mp4": str(rawhand_mp4) if rawhand_mp4.exists() else None,
        "valid_frames": int(valid.sum()),
        "total_frames": int(len(valid)),
        "valid_ratio": float(valid.mean()),
        "invalid_frames": invalid,
    }


def build_single_demo_cmd(args: argparse.Namespace, demo_index: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "phantom.qwenrobot.prepare_egodex_for_phantom",
        "--egodex-root",
        str(args.egodex_root),
        "--output-root",
        str(args.output_root),
        "--demo-index",
        str(demo_index),
        "--demo-name",
        args.demo_name,
        "--demo-num",
        str(demo_index),
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        args.max_frames,
        "--output-fps",
        str(args.output_fps),
        "--retarget",
        args.retarget,
        "--robot",
        args.robot,
        "--gripper",
        args.gripper,
        "--input-resolution",
        str(args.input_resolution),
        "--output-resolution",
        str(args.output_resolution),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.run_hand_inpaint:
        cmd.append("--run-hand-inpaint")
    if args.raw_hand_preview:
        cmd.append("--raw-hand-overlay-preview")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run the EgoDex to Phantom/Masquerade baseline adapter.")
    parser.add_argument("--egodex-root", type=Path, default=DEFAULT_EGODEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=str, default="all", help="all, a comma list, or ranges like 0,2-4")
    parser.add_argument("--demo-name", type=str, default="egodex_phantom")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=str, default="full")
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--robot", type=str, default="Kinova3")
    parser.add_argument("--gripper", type=str, default="Robotiq85")
    parser.add_argument("--retarget", choices=("phantom", "qwen"), default="phantom")
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=256)
    parser.add_argument("--run-hand-inpaint", action="store_true")
    parser.add_argument(
        "--raw-hand-preview",
        action="store_true",
        help="Render an additional robot overlay on the original RGB video for hand-coverage checks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    demos = discover_demos(args.egodex_root)
    indices = parse_indices(args.indices, len(demos))
    summary: dict[str, object] = {"output_root": str(args.output_root), "episodes": []}
    log_dir = args.output_root / "logs"

    for demo_index in indices:
        processed_demo_dir = args.output_root / "processed" / args.demo_name / str(demo_index)
        try:
            cmd = build_single_demo_cmd(args, demo_index)
            run(cmd, log_dir / f"demo_{demo_index}.log")
            export_mp4(
                processed_demo_dir / "video_overlay_Kinova3_shoulders.mkv",
                processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4",
            )
            write_filled_preview(processed_demo_dir, args.output_fps)
            rawhand_mkv = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mkv"
            if rawhand_mkv.exists():
                export_mp4(rawhand_mkv, processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4")
            episode = summarize_demo(processed_demo_dir)
            episode["demo_index"] = demo_index
            episode["source_rel_id"] = demos[demo_index][2]
            summary["episodes"].append(episode)
        except Exception as exc:
            episode = {
                "demo_index": demo_index,
                "source_rel_id": demos[demo_index][2],
                "error": repr(exc),
            }
            summary["episodes"].append(episode)
            (args.output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if not args.continue_on_error:
                raise

    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output_root / "batch_summary.json")


if __name__ == "__main__":
    main()
