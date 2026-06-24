from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapy as media
import numpy as np

from phantom.qwenrobot.prepare_egodex_for_phantom import REPO_ROOT
from phantom.utils.image_utils import convert_video_to_images


@contextmanager
def working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EgoDex arm masks with Phantom's original SAM2 video "
            "propagation style: initialize from one high-quality bbox/keypoint "
            "frame per hand, propagate forward/backward, then union both hands."
        )
    )
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-data",
        choices=("sam2_seed", "bbox"),
        default="sam2_seed",
        help="sam2_seed uses EgoDex shoulder/elbow/wrist/hand boxes; bbox uses Phantom-format hand boxes.",
    )
    parser.add_argument("--point-mode", choices=("original", "all"), default="original")
    parser.add_argument("--close-kernel", type=int, default=7)
    parser.add_argument("--post-dilation", type=int, default=3)
    parser.add_argument("--replace", action="store_true", help="Replace segmentation_processor/masks_arm.npy.")
    return parser.parse_args()


def ensure_original_images(processed_demo_dir: Path) -> Path:
    frames_dir = processed_demo_dir / "original_images"
    if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
        return frames_dir
    if frames_dir.exists():
        for path in frames_dir.glob("*"):
            path.unlink()
    convert_video_to_images(str(processed_demo_dir / "video_L.mp4"), str(frames_dir), square=False)
    return frames_dir


def load_seed_data(processed_demo_dir: Path, seed_data: str) -> np.lib.npyio.NpzFile:
    name = "sam2_seed_data.npz" if seed_data == "sam2_seed" else "bbox_data.npz"
    path = processed_demo_dir / "bbox_processor" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def load_hand_points(processed_demo_dir: Path, side: str, frame_idx: int, point_mode: str) -> np.ndarray:
    hand = np.load(processed_demo_dir / "hand_processor" / f"hand_data_{side}.npz")
    kpts = hand["kpts_2d"][frame_idx].astype(np.float32)
    finite = np.isfinite(kpts).all(axis=1)
    if not finite.any():
        raise ValueError(f"No finite {side} hand points at frame {frame_idx}")
    if point_mode == "original":
        # Phantom's ArmSegmentationProcessor passes np.expand_dims(kpts, axis=1)
        # and one frame index, so DetectorSam2 receives the first keypoint only.
        return kpts[:1]
    return kpts[finite]


def choose_anchor(seed: np.lib.npyio.NpzFile, side: str) -> int | None:
    detected = seed[f"{side}_hand_detected"].astype(bool)
    quality = seed[f"{side}_bbox_min_dist_to_edge"].astype(np.float32)
    valid = detected & (quality > 0.0)
    if not valid.any():
        return None
    masked_quality = np.where(valid, quality, -1.0)
    return int(masked_quality.argmax())


def segments_to_array(segments: dict[int, dict[int, np.ndarray]], n_frames: int, height: int, width: int) -> np.ndarray:
    masks = np.zeros((n_frames, height, width), dtype=bool)
    for frame_idx, objects in segments.items():
        if 0 not in objects:
            continue
        mask = np.asarray(objects[0])
        masks[int(frame_idx)] = mask.reshape(height, width).astype(bool)
    return masks


def process_side(detector, frames_dir: Path, seed: np.lib.npyio.NpzFile, side: str, point_mode: str) -> np.ndarray:
    frame_names = sorted(frames_dir.glob("*.jpg"))
    first = iio.imread(frame_names[0])
    n_frames, height, width = len(frame_names), first.shape[0], first.shape[1]
    anchor = choose_anchor(seed, side)
    if anchor is None:
        return np.zeros((n_frames, height, width), dtype=bool)

    bbox = seed[f"{side}_bboxes"][anchor].astype(np.float32)
    points = load_hand_points(frames_dir.parent, side, anchor, point_mode)
    forward, _ = detector.segment_video(frames_dir, bbox, np.asarray([points]), [anchor], reverse=False)
    reverse, _ = detector.segment_video(frames_dir, bbox, np.asarray([points]), [anchor], reverse=True)
    masks = segments_to_array(forward, n_frames, height, width)
    reverse_masks = segments_to_array(reverse, n_frames, height, width)
    masks[reverse_masks] = True
    return masks


def postprocess(masks: np.ndarray, close_kernel: int, post_dilation: int) -> np.ndarray:
    if close_kernel <= 1 and post_dilation <= 0:
        return masks.astype(bool)
    close = np.ones((close_kernel, close_kernel), dtype=np.uint8) if close_kernel > 1 else None
    dilate = np.ones((post_dilation, post_dilation), dtype=np.uint8) if post_dilation > 0 else None
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        current = mask.astype(np.uint8)
        if close is not None:
            current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, close)
        if dilate is not None:
            current = cv2.dilate(current, dilate, iterations=1)
        out[idx] = current.astype(bool)
    return out


def write_outputs(processed_demo_dir: Path, masks: np.ndarray, replace: bool, summary: dict[str, object]) -> None:
    seg_dir = processed_demo_dir / "segmentation_processor"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out_path = seg_dir / "masks_arm_original_sam2.npy"
    np.save(out_path, masks.astype(np.uint8))

    raw = media.read_video(processed_demo_dir / "video_L.mp4")
    blacked = raw.copy()
    blacked[masks] = 0
    media.write_video(seg_dir / "video_masks_arm_original_sam2.mkv", masks.astype(np.uint8) * 255, fps=15, codec="ffv1")
    media.write_video(seg_dir / "video_sam_arm_original_sam2.mkv", blacked, fps=15, codec="ffv1")

    rough_path = seg_dir / "masks_arm.npy"
    rough = np.load(rough_path).astype(bool) if rough_path.exists() else None
    montage = write_montage(processed_demo_dir, raw, rough, masks)
    if replace:
        backup = seg_dir / "masks_arm.before_original_sam2.npy"
        if rough_path.exists() and not backup.exists():
            np.save(backup, np.load(rough_path))
        np.save(rough_path, masks.astype(np.uint8))

    summary.update(
        {
            "masks_arm_original_sam2": str(out_path),
            "montage": str(montage),
            "mean_mask_area": float(masks.mean()),
            "replaced_masks_arm": bool(replace),
        }
    )
    (seg_dir / "original_sam2_segment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def label(frame: np.ndarray, text: str, idx: int) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, str(idx), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def write_montage(processed_demo_dir: Path, raw: np.ndarray, rough: np.ndarray | None, masks: np.ndarray) -> Path:
    frame_ids = np.linspace(0, len(raw) - 1, min(6, len(raw)), dtype=int).tolist()
    rows = []
    rows.append(np.concatenate([label(raw[idx], "raw", idx) for idx in frame_ids], axis=1))
    if rough is not None:
        rows.append(
            np.concatenate(
                [label(cv2.cvtColor((rough[idx].astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB), "rough_mask", idx) for idx in frame_ids],
                axis=1,
            )
        )
    rows.append(
        np.concatenate(
            [label(cv2.cvtColor((masks[idx].astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB), "original_sam2", idx) for idx in frame_ids],
            axis=1,
        )
    )
    blacked = raw.copy()
    blacked[masks] = 0
    rows.append(np.concatenate([label(blacked[idx], "masked_rgb", idx) for idx in frame_ids], axis=1))
    montage = np.concatenate(rows, axis=0)
    out_path = processed_demo_dir / "segmentation_processor" / "original_sam2_segment_montage.jpg"
    iio.imwrite(out_path, montage, quality=92)
    return out_path


def main() -> None:
    args = parse_args()
    processed_demo_dir = args.processed_demo_dir.resolve()
    frames_dir = ensure_original_images(processed_demo_dir)
    seed = load_seed_data(processed_demo_dir, args.seed_data)

    # DetectorSam2 keeps the upstream relative checkpoint path. Running its
    # constructor from phantom/ preserves the original repository expectation.
    with working_directory(REPO_ROOT / "phantom"):
        from phantom.detectors.detector_sam2 import DetectorSam2

        detector = DetectorSam2()
        left = process_side(detector, frames_dir, seed, "left", args.point_mode)
        right = process_side(detector, frames_dir, seed, "right", args.point_mode)

    masks = postprocess(left | right, args.close_kernel, args.post_dilation)
    write_outputs(
        processed_demo_dir,
        masks,
        args.replace,
        {
            "processed_demo_dir": str(processed_demo_dir),
            "seed_data": args.seed_data,
            "point_mode": args.point_mode,
            "close_kernel": args.close_kernel,
            "post_dilation": args.post_dilation,
        },
    )
    print(processed_demo_dir / "segmentation_processor" / "original_sam2_segment_summary.json")


if __name__ == "__main__":
    main()
