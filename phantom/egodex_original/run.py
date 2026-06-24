from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v3 as iio
import numpy as np

from phantom.qwenrobot.prepare_egodex_for_phantom import (
    DEFAULT_EGODEX_ROOT,
    REPO_ROOT,
    build_phantom_cfg,
    prepare_one_demo,
    run_original_processors,
    run_raw_hand_overlay_preview,
)
from phantom.qwenrobot.batch_prepare_egodex_for_phantom import export_mp4, write_filled_preview


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "phantom_egodex_repro_original"
BASE_OFFSET_ENV_VARS = (
    "PHANTOM_BIMANUAL_BASE0_OFFSET",
    "PHANTOM_BIMANUAL_BASE1_OFFSET",
)


def optional_frame_count(value: str) -> int | None:
    if value.lower() in {"none", "null", "full", "-1", "0"}:
        return None
    parsed = int(value)
    if parsed < 0:
        return None
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EgoDex through the original Phantom/Masquerade fixed-shoulders "
            "Kinova3 pipeline. EgoDex is only adapted into Phantom input files; "
            "retargeting, smoothing, E2FGVI inpainting, and robot overlay use "
            "the original processors."
        )
    )
    parser.add_argument("--egodex-root", type=Path, default=DEFAULT_EGODEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--demo-name", type=str, default="egodex_phantom_original")
    parser.add_argument("--demo-num", type=str, default="0")
    parser.add_argument("--frame-stride", type=int, default=2, help="2 maps 30 FPS EgoDex to Phantom's 15 FPS.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source-video frame to sample.")
    parser.add_argument("--max-frames", type=optional_frame_count, default=None)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=256)
    parser.add_argument("--bimanual-setup", choices=("shoulders", "shoulders1", "shoulders2"), default="shoulders")
    parser.add_argument(
        "--target-center-env",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.25),
        metavar=("X", "Y", "Z"),
        help="Target EgoDex hand-keypoint center for generated EgoDex extrinsics.",
    )
    parser.add_argument(
        "--arm-anchor-alpha",
        type=float,
        default=1.0,
        help="Interpolation from original EPIC extrinsics (0) to EgoDex upper-arm anchored extrinsics (1).",
    )
    extrinsics_group = parser.add_mutually_exclusive_group()
    extrinsics_group.add_argument(
        "--use-original-epic-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="original_epic_shoulders",
        default="original_epic_shoulders",
        help="Use Phantom/Masquerade's fixed EPIC shoulders camera extrinsics.",
    )
    extrinsics_group.add_argument(
        "--use-generated-egodex-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="generated_egodex_target_center",
        help="Generate EgoDex extrinsics that center EgoDex hand keypoints in the Phantom workspace.",
    )
    extrinsics_group.add_argument(
        "--use-arm-anchor-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="arm_anchor",
        help="Align EgoDex upper-arm anchors to Phantom's fixed shoulders bases.",
    )
    extrinsics_group.add_argument(
        "--use-root-translation-extrinsics",
        dest="extrinsics_mode",
        action="store_const",
        const="root_translation",
        help="Keep Phantom's original EPIC rotation and translate fixed shoulders roots toward EgoDex arm anchors.",
    )
    parser.add_argument(
        "--root-anchor-key",
        choices=("Shoulder", "Arm", "Forearm"),
        default="Arm",
        help="EgoDex body anchor used by --use-root-translation-extrinsics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-hand-inpaint",
        action="store_true",
        help="Only useful for fast debugging; the original reproduction runs E2FGVI by default.",
    )
    parser.add_argument("--skip-robot-inpaint", action="store_true")
    parser.add_argument("--skip-raw-hand-preview", action="store_true")
    parser.add_argument("--skip-mp4-export", action="store_true")
    parser.add_argument(
        "--allow-base-offset-env",
        action="store_true",
        help="Allow PHANTOM_BIMANUAL_BASE*_OFFSET to affect the robosuite shoulders roots.",
    )
    return parser.parse_args()


def make_legacy_prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        egodex_root=args.egodex_root,
        output_root=args.output_root,
        demo_index=args.demo_index,
        demo_name=args.demo_name,
        demo_num=args.demo_num,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        output_fps=args.output_fps,
        robot="Kinova3",
        gripper="Robotiq85",
        bimanual_setup=args.bimanual_setup,
        retarget="phantom",
        input_resolution=args.input_resolution,
        output_resolution=args.output_resolution,
        target_center_env=tuple(args.target_center_env),
        arm_anchor_alpha=args.arm_anchor_alpha,
        root_anchor_key=args.root_anchor_key,
        extrinsics_mode=args.extrinsics_mode,
        use_original_epic_extrinsics=args.extrinsics_mode == "original_epic_shoulders",
        run_hand_inpaint=not args.skip_hand_inpaint,
        reuse_hand_inpaint=None,
        skip_robot_inpaint=args.skip_robot_inpaint,
        raw_hand_overlay_preview=not args.skip_raw_hand_preview,
        overwrite=args.overwrite,
        prepare_only=False,
    )


def copy_manifest_to_processed(raw_demo_dir: Path, processed_demo_dir: Path) -> None:
    raw_manifest = raw_demo_dir / "adapter_manifest.json"
    if raw_manifest.exists():
        shutil.copy2(raw_manifest, processed_demo_dir / "adapter_manifest.json")


def read_frame(video_path: Path, frame_idx: int, fallback_shape: tuple[int, int] = (256, 456)) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def put_label(frame: np.ndarray, label: str, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, str(frame_idx), (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def make_diagnostic_montage(processed_demo_dir: Path) -> Path:
    overlay_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4"
    rawhand_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4"
    clean_mkv = processed_demo_dir / "inpaint_processor" / "video_human_inpaint.mkv"
    raw_mp4 = processed_demo_dir / "video_L.mp4"
    masks_path = processed_demo_dir / "segmentation_processor" / "masks_arm.npy"

    cap = cv2.VideoCapture(str(raw_mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 256
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 456
    cap.release()
    if total <= 0:
        raise RuntimeError(f"Cannot inspect video frame count: {raw_mp4}")

    frame_ids = np.linspace(0, total - 1, min(6, total), dtype=int).tolist()
    rows: list[np.ndarray] = []
    for label, video_path in (
        ("raw", raw_mp4),
        ("clean_e2fgvi", clean_mkv),
        ("overlay_rawhand", rawhand_mp4),
        ("overlay_e2fgvi", overlay_mp4),
    ):
        if not video_path.exists():
            continue
        rows.append(
            np.concatenate(
                [put_label(read_frame(video_path, idx, (height, width)), label, idx) for idx in frame_ids],
                axis=1,
            )
        )

    if masks_path.exists():
        masks = np.load(masks_path)
        mask_frames = []
        for idx in frame_ids:
            mask = (masks[idx].astype(np.uint8) * 255)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_frames.append(put_label(mask_rgb, "mask_arm", idx))
        rows.insert(1, np.concatenate(mask_frames, axis=1))

    montage = np.concatenate(rows, axis=0)
    output_path = processed_demo_dir / "phantom_original_repro_montage.jpg"
    iio.imwrite(output_path, montage, quality=92)
    return output_path


def summarize(processed_demo_dir: Path, montage: Path | None) -> dict[str, object]:
    overlay_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4"
    rawhand_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4"
    filled_preview_mp4 = processed_demo_dir / "video_overlay_Kinova3_shoulders_preview_filled.mp4"
    training_path = processed_demo_dir / "inpaint_processor" / "training_data_shoulders.npz"
    valid = np.load(training_path, allow_pickle=True)["valid"].astype(bool) if training_path.exists() else None
    summary: dict[str, object] = {
        "processed_demo_dir": str(processed_demo_dir),
        "overlay_mp4": str(overlay_mp4) if overlay_mp4.exists() else None,
        "rawhand_overlay_mp4": str(rawhand_mp4) if rawhand_mp4.exists() else None,
        "filled_preview_mp4": str(filled_preview_mp4) if filled_preview_mp4.exists() else None,
        "montage": str(montage) if montage is not None else None,
    }
    if valid is not None:
        summary.update(
            {
                "valid_frames": int(valid.sum()),
                "total_frames": int(len(valid)),
                "valid_ratio": float(valid.mean()),
                "invalid_frames": np.where(~valid)[0].astype(int).tolist(),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if not args.allow_base_offset_env:
        for env_name in BASE_OFFSET_ENV_VARS:
            os.environ.pop(env_name, None)
    legacy_args = make_legacy_prepare_args(args)

    raw_demo_dir, intrinsics_path, extrinsics_path = prepare_one_demo(legacy_args)
    cfg = build_phantom_cfg(legacy_args, intrinsics_path, extrinsics_path)
    run_original_processors(
        cfg,
        args.demo_num,
        include_robot_inpaint=not args.skip_robot_inpaint,
        retarget="phantom",
        run_hand_inpaint=not args.skip_hand_inpaint,
        reuse_hand_inpaint=None,
    )

    processed_demo_dir = args.output_root / "processed" / args.demo_name / args.demo_num
    copy_manifest_to_processed(raw_demo_dir, processed_demo_dir)

    if not args.skip_robot_inpaint and not args.skip_raw_hand_preview:
        run_raw_hand_overlay_preview(cfg, args.demo_num, args.overwrite)

    if not args.skip_mp4_export:
        overlay_mkv = processed_demo_dir / "video_overlay_Kinova3_shoulders.mkv"
        if overlay_mkv.exists():
            export_mp4(overlay_mkv, processed_demo_dir / "video_overlay_Kinova3_shoulders.mp4")
        rawhand_mkv = processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mkv"
        if rawhand_mkv.exists():
            export_mp4(rawhand_mkv, processed_demo_dir / "video_overlay_Kinova3_shoulders_rawhand.mp4")
        if (processed_demo_dir / "inpaint_processor" / "training_data_shoulders.npz").exists():
            valid = np.load(
                processed_demo_dir / "inpaint_processor" / "training_data_shoulders.npz",
                allow_pickle=True,
            )["valid"].astype(bool)
            if valid.any():
                write_filled_preview(processed_demo_dir, args.output_fps)

    montage = None
    if not args.skip_robot_inpaint:
        montage = make_diagnostic_montage(processed_demo_dir)

    summary = summarize(processed_demo_dir, montage)
    summary_path = processed_demo_dir / "phantom_original_repro_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
