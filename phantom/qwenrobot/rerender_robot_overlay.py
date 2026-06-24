from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("MUJOCO_GL", "egl")

from phantom.qwenrobot.gripper_frame_utils import recover_qwen_rotations_from_legacy
from phantom.qwenrobot.prepare_egodex_for_phantom import add_phantom_submodules_to_path, build_phantom_cfg


def run(cmd: list[str]) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def validate_video_file(path: Path, *, min_mean: float = 1.0) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video is not readable: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {path}")
    sample_ids = sorted({0, frame_count // 2, frame_count - 1})
    means = []
    for frame_id in sample_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not decode frame {frame_id} from {path}")
        means.append(float(frame.mean()))
    cap.release()
    if max(means) < min_mean:
        raise RuntimeError(f"Video appears all-black: {path}, sampled means={means}")


def output_root_from_processed(processed_demo_dir: Path) -> tuple[Path, str, str]:
    processed_demo_dir = processed_demo_dir.resolve()
    demo_num = processed_demo_dir.name
    demo_name = processed_demo_dir.parent.name
    processed_root = processed_demo_dir.parent.parent
    if processed_root.name != "processed":
        raise ValueError(f"Expected .../processed/<demo>/<num>, got {processed_demo_dir}")
    return processed_root.parent, demo_name, demo_num


def find_intrinsics(output_root: Path, demo_name: str, demo_num: str) -> Path:
    candidates = [
        output_root / "raw" / demo_name / demo_num / "egodex_camera_intrinsics.json",
        output_root / "egodex_camera_intrinsics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No EgoDex intrinsics found in: {candidates}")


def prepare_scratch_demo(
    processed_demo_dir: Path,
    inpaint_video: Path,
    *,
    output_root: Path,
    demo_name: str,
    demo_num: str,
    suffix: str,
    overwrite: bool,
    depth_path: Path | None,
) -> tuple[str, Path]:
    scratch_demo_name = f"{demo_name}_{suffix}_rerender"
    scratch_processed_dir = output_root / "processed" / scratch_demo_name / demo_num
    scratch_raw_dir = output_root / "raw" / scratch_demo_name / demo_num
    if overwrite:
        for path in (scratch_processed_dir, scratch_raw_dir):
            if path.exists():
                shutil.rmtree(path)
    scratch_processed_dir.parent.mkdir(parents=True, exist_ok=True)
    scratch_raw_dir.parent.mkdir(parents=True, exist_ok=True)
    if not scratch_processed_dir.exists():
        shutil.copytree(processed_demo_dir, scratch_processed_dir)
    scratch_raw_dir.mkdir(parents=True, exist_ok=True)

    target_inpaint = scratch_processed_dir / "inpaint_processor" / "video_human_inpaint.mkv"
    target_inpaint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inpaint_video, target_inpaint)
    if depth_path is not None:
        shutil.copy2(depth_path, scratch_processed_dir / "depth.npy")
    return scratch_demo_name, scratch_processed_dir


def apply_qwen_tool_axis_fix(processed_demo_dir: Path, bimanual_setup: str) -> None:
    qwen_to_phantom_tool = (
        Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
        @ Rotation.from_euler("z", -135.0, degrees=True).as_matrix()
    )
    candidates = []
    for folder, stem in (
        ("action_processor", "actions"),
        ("smoothing_processor", "smoothed_actions"),
    ):
        for side in ("left", "right"):
            candidates.append(processed_demo_dir / folder / f"{stem}_{side}_{bimanual_setup}.npz")

    for path in candidates:
        if not path.exists():
            continue
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        if "ee_oris" not in payload:
            continue
        payload["ee_oris"] = np.asarray(payload["ee_oris"]) @ qwen_to_phantom_tool
        np.savez(path, **payload)


def apply_tool_roll(processed_demo_dir: Path, bimanual_setup: str, tool_roll_deg: float) -> None:
    if abs(tool_roll_deg) < 1e-12:
        return
    tool_roll = Rotation.from_euler("x", tool_roll_deg, degrees=True).as_matrix()
    candidates = []
    for folder, stem in (
        ("action_processor", "actions"),
        ("smoothing_processor", "smoothed_actions"),
    ):
        for side in ("left", "right"):
            candidates.append(processed_demo_dir / folder / f"{stem}_{side}_{bimanual_setup}.npz")

    for path in candidates:
        if not path.exists():
            continue
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        if "ee_oris" not in payload:
            continue
        payload["ee_oris"] = np.asarray(payload["ee_oris"]) @ tool_roll
        np.savez(path, **payload)


def apply_gripper_frame_calibration(
    processed_demo_dir: Path,
    bimanual_setup: str,
    *,
    recover_qwen_from_legacy: bool,
    gripper_frame_matrix: np.ndarray | None,
) -> None:
    if not recover_qwen_from_legacy and gripper_frame_matrix is None:
        return
    candidates = []
    for folder, stem in (
        ("action_processor", "actions"),
        ("smoothing_processor", "smoothed_actions"),
    ):
        for side in ("left", "right"):
            candidates.append(processed_demo_dir / folder / f"{stem}_{side}_{bimanual_setup}.npz")

    for path in candidates:
        if not path.exists():
            continue
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        if "ee_oris" not in payload:
            continue
        rotations = np.asarray(payload["ee_oris"], dtype=np.float64)
        if recover_qwen_from_legacy:
            rotations = recover_qwen_rotations_from_legacy(rotations)
        if gripper_frame_matrix is not None:
            rotations = rotations @ gripper_frame_matrix
        payload["ee_oris"] = rotations
        np.savez(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender Phantom robot overlay with a supplied inpainted background video.")
    parser.add_argument("--processed-demo-dir", type=Path, required=True)
    parser.add_argument("--inpaint-video", type=Path, required=True)
    parser.add_argument("--camera-intrinsics", type=Path, default=None)
    parser.add_argument("--camera-extrinsics", type=Path, default=Path("/mnt/project_rlinf/jlchen/code/phantom_reference/phantom/camera/camera_extrinsics_ego_bimanual_shoulders.json"))
    parser.add_argument("--robot", type=str, default="Kinova3")
    parser.add_argument("--gripper", type=str, default="Robotiq85")
    parser.add_argument("--bimanual-setup", type=str, default="shoulders")
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=256)
    parser.add_argument("--suffix", type=str, default="propainter")
    parser.add_argument("--depth-for-overlay", action="store_true")
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--depth-occlusion-margin", type=float, default=0.03)
    parser.add_argument("--base-search-result", type=Path, default=None)
    parser.add_argument("--qwen-tool-axis-fix", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-mp4", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_phantom_submodules_to_path()
    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

    processed_demo_dir = args.processed_demo_dir.resolve()
    inpaint_video = args.inpaint_video.resolve()
    if not processed_demo_dir.exists():
        raise FileNotFoundError(processed_demo_dir)
    if not inpaint_video.exists():
        raise FileNotFoundError(inpaint_video)

    output_root, demo_name, demo_num = output_root_from_processed(processed_demo_dir)
    intrinsics = args.camera_intrinsics.resolve() if args.camera_intrinsics else find_intrinsics(output_root, demo_name, demo_num)
    extrinsics = args.camera_extrinsics.resolve()
    depth_path = args.depth_path.resolve() if args.depth_path is not None else None
    if args.depth_for_overlay and depth_path is None:
        default_depth = processed_demo_dir / "depth.npy"
        if default_depth.exists():
            depth_path = default_depth
        else:
            raise FileNotFoundError("--depth-for-overlay requires --depth-path or processed_demo_dir/depth.npy")
    scratch_demo_name, scratch_processed_dir = prepare_scratch_demo(
        processed_demo_dir,
        inpaint_video,
        output_root=output_root,
        demo_name=demo_name,
        demo_num=demo_num,
        suffix=args.suffix,
        overwrite=args.overwrite,
        depth_path=depth_path,
    )
    if args.qwen_tool_axis_fix:
        apply_qwen_tool_axis_fix(scratch_processed_dir, args.bimanual_setup)
    base_payload = None
    if args.base_search_result is not None:
        print(
            "WARNING: --base-search-result in rerender_robot_overlay applies legacy Phantom/robosuite "
            "shoulders offsets. Direct MuJoCo results from search_base/render_robot do not use this path.",
            flush=True,
        )
        base_payload = json.loads(args.base_search_result.resolve().read_text(encoding="utf-8"))
        os.environ["PHANTOM_BIMANUAL_BASE0_OFFSET"] = ",".join(str(v) for v in base_payload["base0_offset"])
        os.environ["PHANTOM_BIMANUAL_BASE1_OFFSET"] = ",".join(str(v) for v in base_payload["base1_offset"])
        if "global_yaw_deg" in base_payload:
            os.environ["PHANTOM_BIMANUAL_GLOBAL_YAW_DEG"] = str(base_payload["global_yaw_deg"])
        matrix_payload = base_payload.get("gripper_frame_matrix")
        gripper_frame_matrix = None
        if matrix_payload is not None:
            gripper_frame_matrix = np.asarray(matrix_payload, dtype=np.float64)
            if gripper_frame_matrix.shape != (3, 3):
                raise ValueError(f"gripper_frame_matrix must be 3x3, got {gripper_frame_matrix.shape}")
        apply_gripper_frame_calibration(
            scratch_processed_dir,
            args.bimanual_setup,
            recover_qwen_from_legacy=bool(base_payload.get("recover_qwen_from_legacy", False)),
            gripper_frame_matrix=gripper_frame_matrix,
        )
        apply_tool_roll(scratch_processed_dir, args.bimanual_setup, float(base_payload.get("tool_roll_deg", 0.0)))
    cfg_args = Namespace(
        output_root=output_root,
        demo_name=scratch_demo_name,
        demo_num=demo_num,
        input_resolution=args.input_resolution,
        output_resolution=args.output_resolution,
        robot=args.robot,
        gripper=args.gripper,
        bimanual_setup=args.bimanual_setup,
        depth_for_overlay=args.depth_for_overlay,
        depth_occlusion_margin=args.depth_occlusion_margin,
    )
    cfg = build_phantom_cfg(cfg_args, intrinsics, extrinsics)
    try:
        RobotInpaintProcessor(cfg).process_one_demo(demo_num)
    finally:
        if args.base_search_result is not None:
            os.environ.pop("PHANTOM_BIMANUAL_BASE0_OFFSET", None)
            os.environ.pop("PHANTOM_BIMANUAL_BASE1_OFFSET", None)
            os.environ.pop("PHANTOM_BIMANUAL_GLOBAL_YAW_DEG", None)

    stem = f"video_overlay_{args.robot}_{args.bimanual_setup}"
    source_mkv = scratch_processed_dir / f"{stem}.mkv"
    source_training = scratch_processed_dir / "inpaint_processor" / f"training_data_{args.bimanual_setup}.npz"
    target_mkv = processed_demo_dir / f"{stem}_{args.suffix}.mkv"
    target_training = processed_demo_dir / "inpaint_processor" / f"training_data_shoulders_{args.suffix}.npz"
    shutil.copy2(source_mkv, target_mkv)
    shutil.copy2(source_training, target_training)
    validate_video_file(target_mkv)
    source_debug = scratch_processed_dir / "inpaint_processor" / "depth_overlay_debug"
    if source_debug.exists():
        target_debug = processed_demo_dir / "inpaint_processor" / f"depth_overlay_debug_{args.suffix}"
        if target_debug.exists():
            shutil.rmtree(target_debug)
        shutil.copytree(source_debug, target_debug)
        print(f"depth_overlay_debug={target_debug}")
    print(f"overlay_mkv={target_mkv}")
    print(f"training_data={target_training}")

    if args.export_mp4:
        target_mp4 = processed_demo_dir / f"{stem}_{args.suffix}.mp4"
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                target_mkv,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                target_mp4,
            ]
        )
        validate_video_file(target_mp4)
        print(f"overlay_mp4={target_mp4}")


if __name__ == "__main__":
    main()
