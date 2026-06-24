from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DEMO = (
    REPO_ROOT
    / "outputs"
    / "phantom_egodex_exact_epic_full_qwen_action"
    / "processed"
    / "egodex_phantom"
    / "0"
)
DEFAULT_KINOVA_DIR = REPO_ROOT / "outputs" / "kinova_camera_ik_qwen_action_pad120_d35_w_mid_masked"
DEFAULT_CLEAN_VIDEO = (
    REPO_ROOT
    / "outputs"
    / "phantom_egodex_exact_epic_full"
    / "processed"
    / "egodex_phantom"
    / "0"
    / "inpaint_processor"
    / "video_human_inpaint_propainter_sam31_strict_phantom_dilate.mkv"
)
DEFAULT_SAM_MASK = (
    REPO_ROOT
    / "outputs"
    / "phantom_egodex_exact_epic_full"
    / "processed"
    / "egodex_phantom"
    / "0"
    / "segmentation_processor"
    / "masks_arm_sam31_strict_phantom_dilate.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "qwenrobot_egodex_episode0_shoulder" / "08_upstream_phantom_check"


def video_probe(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        means = []
        for frame_id in sorted({0, max(0, frames // 2), max(0, frames - 1)}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if ok:
                means.append(float(frame.mean()))
    finally:
        cap.release()
    if frames <= 0:
        raise RuntimeError(f"Video has no frames: {path}")
    if means and max(means) < 1.0:
        raise RuntimeError(f"Video appears black: {path}, means={means}")
    return {"frames": frames, "width": width, "height": height, "fps": fps, "sample_means": means}


def load_robot_masks(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    key = "robot_mask" if "robot_mask" in data.files else data.files[0]
    return data[key].astype(bool)


def write_streamed_replacement(
    *,
    clean_video: Path,
    robot_video: Path,
    robot_mask: Path,
    output_video: Path,
    montage: Path,
) -> dict[str, object]:
    masks = load_robot_masks(robot_mask)
    clean_cap = cv2.VideoCapture(str(clean_video))
    robot_cap = cv2.VideoCapture(str(robot_video))
    if not clean_cap.isOpened():
        raise RuntimeError(f"Cannot open clean video: {clean_video}")
    if not robot_cap.isOpened():
        raise RuntimeError(f"Cannot open robot video: {robot_video}")

    fps = float(clean_cap.get(cv2.CAP_PROP_FPS)) or float(robot_cap.get(cv2.CAP_PROP_FPS)) or 15.0
    width = int(clean_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(clean_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {output_video}")

    total = int(min(
        clean_cap.get(cv2.CAP_PROP_FRAME_COUNT),
        robot_cap.get(cv2.CAP_PROP_FRAME_COUNT),
        len(masks),
    ))
    sample_ids = set(np.linspace(0, max(total - 1, 0), min(5, max(total, 1)), dtype=int).tolist())
    montage_frames: list[np.ndarray] = []
    robot_area = []
    written = 0
    try:
        for frame_idx in range(total):
            ok_clean, clean = clean_cap.read()
            ok_robot, robot = robot_cap.read()
            if not ok_clean or not ok_robot:
                break
            if robot.shape[:2] != (height, width):
                robot = cv2.resize(robot, (width, height), interpolation=cv2.INTER_AREA)
            mask = masks[frame_idx]
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            out = clean.copy()
            out[mask] = robot[mask]
            writer.write(out)
            robot_area.append(float(mask.mean()))
            if frame_idx in sample_ids:
                labeled = out.copy()
                cv2.putText(
                    labeled,
                    f"upstream_phantom:f{frame_idx}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                montage_frames.append(labeled)
            written += 1
    finally:
        clean_cap.release()
        robot_cap.release()
        writer.release()

    if montage_frames:
        montage.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(montage), np.concatenate(montage_frames, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return {
        "frames": written,
        "fps": fps,
        "width": width,
        "height": height,
        "mean_robot_area_ratio": float(np.mean(robot_area)) if robot_area else 0.0,
        "max_robot_area_ratio": float(np.max(robot_area)) if robot_area else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the jigsaw replacement from upstream Phantom assets: exported real "
            "Kinova3 XML robot RGB, robot masks, and ProPainter/SAM cleaned background. "
            "This intentionally bypasses the rejected qwenrobot procedural/proxy renderer."
        )
    )
    parser.add_argument("--processed-demo-dir", type=Path, default=DEFAULT_PROCESSED_DEMO)
    parser.add_argument("--kinova-dir", type=Path, default=DEFAULT_KINOVA_DIR)
    parser.add_argument("--clean-video", type=Path, default=DEFAULT_CLEAN_VIDEO)
    parser.add_argument("--sam-mask", type=Path, default=DEFAULT_SAM_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="extra__assemble_disassemble_jigsaw_puzzle__0")
    parser.add_argument("--overwrite-selected-montage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{args.output_name}_upstream_phantom_replacement.mp4"
    montage = output_dir / f"{args.output_name}.jpg"
    robot_video = args.kinova_dir.resolve() / "video_robot_Kinova3_camera_ik.mp4"
    robot_mask = args.kinova_dir.resolve() / "robot_masks_Kinova3_camera_ik.npz"

    stream_stats = write_streamed_replacement(
        clean_video=args.clean_video.resolve(),
        robot_video=robot_video,
        robot_mask=robot_mask,
        output_video=output_video,
        montage=montage,
    )

    selected_montage = (
        REPO_ROOT / "outputs" / "qwenrobot_egodex_episode0_shoulder" / "04_montage" / f"{args.output_name}.jpg"
    )
    backup = selected_montage.with_suffix(".bad_proxy_backup.jpg")
    if args.overwrite_selected_montage:
        selected_montage.parent.mkdir(parents=True, exist_ok=True)
        if selected_montage.exists() and not backup.exists():
            shutil.copy2(selected_montage, backup)
        shutil.copy2(montage, selected_montage)

    ik_manifest_path = args.kinova_dir.resolve() / "kinova_camera_ik_manifest.json"
    ik_manifest = json.loads(ik_manifest_path.read_text(encoding="utf-8")) if ik_manifest_path.exists() else {}
    summary = {
        "stage": "upstream_phantom_real_robot_replacement",
        "renderer": "exported Phantom Kinova3 XML + MuJoCo camera-space IK",
        "source_repo": "https://github.com/MarionLepert/phantom",
        "processed_demo_dir": str(args.processed_demo_dir.resolve()),
        "clean_video": str(args.clean_video.resolve()),
        "sam_mask": str(args.sam_mask.resolve()),
        "robot_video": str(robot_video),
        "robot_mask": str(robot_mask),
        "output_video": str(output_video),
        "montage": str(montage),
        "selected_montage": str(selected_montage) if args.overwrite_selected_montage else None,
        "bad_proxy_backup": str(backup) if backup.exists() else None,
        "video_probe": video_probe(output_video),
        "stream_composite": stream_stats,
        "mean_left_ik_error_m": ik_manifest.get("mean_left_ik_error"),
        "mean_right_ik_error_m": ik_manifest.get("mean_right_ik_error"),
        "failed_left_frames": ik_manifest.get("failed_left_frames"),
        "failed_right_frames": ik_manifest.get("failed_right_frames"),
        "robosuite_processor_note": (
            "Full RobotInpaintProcessor was checked with OSMesa, but this machine killed "
            "the process during offscreen env initialization; this entry uses the already "
            "exported real Phantom Kinova XML assets instead of proxy geometry."
        ),
    }
    manifest_path = output_dir / "upstream_phantom_final_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
