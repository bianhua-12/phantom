from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_offset(value: str) -> tuple[np.ndarray, np.ndarray, str]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) == 3:
        base0 = base1 = np.asarray(parts, dtype=np.float64)
    elif len(parts) == 6:
        base0 = np.asarray(parts[:3], dtype=np.float64)
        base1 = np.asarray(parts[3:], dtype=np.float64)
    else:
        raise argparse.ArgumentTypeError("offset must be x,y,z or base0x,base0y,base0z,base1x,base1y,base1z")
    label = "_".join(f"{part:+.2f}" for part in parts).replace("+", "p").replace("-", "m").replace(".", "d")
    return base0, base1, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Phantom bimanual base offsets on a short EgoDex clip.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "_base_offset_search")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--offset",
        type=parse_offset,
        action="append",
        required=True,
        help="Candidate offset. Use x,y,z for both bases or six values for base0 and base1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--montage", type=Path, default=None)
    return parser.parse_args()


def run_candidate(args: argparse.Namespace, base0: np.ndarray, base1: np.ndarray, label: str) -> Path:
    candidate_root = args.output_root / label
    if args.overwrite and candidate_root.exists():
        shutil.rmtree(candidate_root)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PHANTOM_BIMANUAL_BASE0_OFFSET"] = ",".join(f"{x:.6g}" for x in base0)
    env["PHANTOM_BIMANUAL_BASE1_OFFSET"] = ",".join(f"{x:.6g}" for x in base1)
    cmd = [
        str(args.python),
        "-m",
        "phantom.egodex_original.run",
        "--output-root",
        str(candidate_root),
        "--demo-index",
        str(args.demo_index),
        "--demo-name",
        "egodex_probe",
        "--demo-num",
        "0",
        "--frame-stride",
        str(args.frame_stride),
        "--start-frame",
        str(args.start_frame),
        "--max-frames",
        str(args.max_frames),
        "--skip-hand-inpaint",
        "--skip-raw-hand-preview",
        "--allow-base-offset-env",
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return candidate_root / "processed" / "egodex_probe" / "0"


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return np.asarray(frames, dtype=np.uint8)


def score_candidate(processed_dir: Path) -> dict[str, object]:
    video_path = processed_dir / "video_overlay_Kinova3_shoulders.mp4"
    raw_path = processed_dir / "video_rgb_imgs.mkv"
    mask_path = processed_dir / "segmentation_processor" / "masks_arm.npy"
    training_path = processed_dir / "inpaint_processor" / "training_data_shoulders.npz"

    valid = np.load(training_path, allow_pickle=True)["valid"].astype(bool)
    overlay = read_video(video_path)
    raw = read_video(raw_path)
    masks = np.load(mask_path).astype(bool)
    n = min(len(overlay), len(raw), len(masks), len(valid))
    overlay, raw, masks, valid = overlay[:n], raw[:n], masks[:n], valid[:n]
    robot_mask = np.linalg.norm(overlay.astype(np.int16) - raw.astype(np.int16), axis=-1) > 18.0
    robot_mask &= valid[:, None, None]
    masks &= valid[:, None, None]

    intersection = np.logical_and(robot_mask, masks).sum()
    robot_area = robot_mask.sum()
    human_area = masks.sum()
    precision = float(intersection / max(robot_area, 1))
    coverage = float(intersection / max(human_area, 1))
    f1 = float(2 * precision * coverage / max(precision + coverage, 1e-9))
    valid_ratio = float(valid.mean()) if len(valid) else 0.0
    return {
        "processed_dir": str(processed_dir),
        "video": str(video_path),
        "valid_frames": int(valid.sum()),
        "total_frames": int(n),
        "valid_ratio": valid_ratio,
        "robot_pixels": int(robot_area),
        "human_mask_pixels": int(human_area),
        "intersection_pixels": int(intersection),
        "precision": precision,
        "coverage": coverage,
        "f1": f1,
        "adjusted_f1": float(f1 * valid_ratio),
    }


def label_frame(frame: np.ndarray, label: str, metrics: dict[str, object], frame_idx: int) -> np.ndarray:
    out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (260, 54), (0, 0, 0), -1)
    cv2.putText(out, label[:24], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    text = f"adj={metrics['adjusted_f1']:.3f} cov={metrics['coverage']:.3f} v={metrics['valid_frames']}/{metrics['total_frames']} f{frame_idx}"
    cv2.putText(out, text, (6, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def write_montage(path: Path, scored: list[tuple[str, dict[str, object]]]) -> None:
    rows = []
    frame_ids = [0, 4, 8, 11]
    for label, metrics in scored:
        frames = read_video(Path(metrics["video"]))
        cells = []
        for idx in frame_ids:
            if idx >= len(frames):
                continue
            cells.append(label_frame(frames[idx], label, metrics, idx))
        if cells:
            rows.append(np.concatenate(cells, axis=1))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rows, axis=0), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    scored: list[tuple[str, dict[str, object]]] = []
    for base0, base1, label in args.offset:
        processed_dir = run_candidate(args, base0, base1, label)
        metrics = score_candidate(processed_dir)
        metrics["base0_offset"] = base0.tolist()
        metrics["base1_offset"] = base1.tolist()
        scored.append((label, metrics))
        print(json.dumps({label: metrics}, indent=2), flush=True)

    summary = {label: metrics for label, metrics in scored}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.montage:
        write_montage(args.montage, scored)
        print(f"montage={args.montage}")


if __name__ == "__main__":
    main()
