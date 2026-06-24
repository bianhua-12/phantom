from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, check=True)


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def prepare_inputs(
    processed_demo_dir: Path,
    work_dir: Path,
    *,
    mask_path: Path | None,
    start_frame: int,
    max_frames: int | None,
) -> tuple[Path, Path, int, int, int]:
    video_path = processed_demo_dir / "video_rgb_imgs.mkv"
    mask_path = mask_path or processed_demo_dir / "segmentation_processor" / "masks_arm.npy"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    frames_dir = work_dir / "frames"
    masks_dir = work_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    masks = np.load(mask_path)
    total = min(source_frames, len(masks))
    end = total if max_frames is None else min(total, start_frame + max_frames)
    if start_frame < 0 or start_frame >= end:
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end}, total={total}")

    for old in frames_dir.glob("*.png"):
        old.unlink()
    for old in masks_dir.glob("*.png"):
        old.unlink()

    for out_i, src_i in enumerate(range(start_frame, end)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_i)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {src_i} from {video_path}")
        cv2.imwrite(str(frames_dir / f"{out_i:06d}.png"), frame_bgr)
        mask = masks[src_i].astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"{out_i:06d}.png"), mask * 255)
    cap.release()
    return frames_dir, masks_dir, end - start_frame, width, height


def transcode_to_mkv(input_mp4: Path, output_mkv: Path, width: int, height: int, fps: float) -> None:
    output_mkv.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            input_mp4,
            "-vf",
            f"scale={width}:{height}",
            "-r",
            str(fps),
            "-c:v",
            "ffv1",
            output_mkv,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProPainter on an existing Phantom/EgoDex processed demo.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--propainter-root", type=Path, default=Path("/mnt/project_rlinf/jlchen/code/ProPainter"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    propainter_root = args.propainter_root.resolve()
    if not (propainter_root / "inference_propainter.py").exists():
        raise FileNotFoundError(propainter_root / "inference_propainter.py")

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else processed_demo_dir / "propainter_processor" / f"frames_{args.start_frame}_{args.max_frames or 'end'}"
    )
    output_video = (
        args.output_video.resolve()
        if args.output_video is not None
        else processed_demo_dir / "inpaint_processor" / "video_human_inpaint_propainter.mkv"
    )
    frames_dir, masks_dir, n_frames, width, height = prepare_inputs(
        processed_demo_dir,
        work_dir,
        mask_path=args.mask_path.resolve() if args.mask_path is not None else None,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    result_root = work_dir / "results"
    cmd = [
        args.python,
        propainter_root / "inference_propainter.py",
        "--video",
        frames_dir,
        "--mask",
        masks_dir,
        "--output",
        result_root,
        "--height",
        str(height),
        "--width",
        str(width),
        "--save_fps",
        str(int(round(args.fps))),
        "--mask_dilation",
        str(args.mask_dilation),
        "--subvideo_length",
        str(args.subvideo_length),
        "--neighbor_length",
        str(args.neighbor_length),
        "--ref_stride",
        str(args.ref_stride),
        "--raft_iter",
        str(args.raft_iter),
    ]
    if args.fp16:
        cmd.append("--fp16")
    run(cmd, cwd=propainter_root)

    inpaint_mp4 = result_root / frames_dir.name / "inpaint_out.mp4"
    if not inpaint_mp4.exists():
        raise FileNotFoundError(inpaint_mp4)
    transcode_to_mkv(inpaint_mp4, output_video, width, height, args.fps)
    print(f"propainter_inpaint={output_video}")
    print(f"frames={n_frames}")


if __name__ == "__main__":
    main()
