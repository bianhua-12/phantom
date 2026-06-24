from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.qwenrobot.gripper_frame_utils import (
    apply_frame_candidate,
    recover_qwen_rotations_from_legacy,
    signed_permutation_candidates,
)
from phantom.qwenrobot.prepare_egodex_for_phantom import add_phantom_submodules_to_path, build_phantom_cfg
from phantom.qwenrobot.rerender_robot_overlay import find_intrinsics, output_root_from_processed
from phantom.qwenrobot.search_phantom_base_offsets import (
    clear_base_env,
    load_hand_centers,
    load_smoothed,
    set_base_env,
    side_projection_metrics,
    visual_metrics,
)


DEFAULT_FRAMES = [23, 47, 71, 95, 119]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Qwen-to-Phantom gripper frame conversion.")
    parser.add_argument(
        "--processed-demo-dir",
        type=Path,
        default=Path("outputs/phantom_egodex_exact_epic/processed/egodex_phantom/0"),
    )
    parser.add_argument("--camera-extrinsics", type=Path, default=Path("phantom/camera/camera_extrinsics_ego_bimanual_shoulders.json"))
    parser.add_argument("--base-search-result", type=Path, required=True)
    parser.add_argument("--robot", type=str, default="Kinova3")
    parser.add_argument("--gripper", type=str, default="Robotiq85")
    parser.add_argument("--bimanual-setup", type=str, default="shoulders")
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=256)
    parser.add_argument("--frames", type=int, nargs="*", default=DEFAULT_FRAMES)
    parser.add_argument("--tracking-threshold", type=float, default=0.10)
    parser.add_argument(
        "--rotation-source",
        choices=("recover-qwen-from-legacy", "as-is"),
        default="recover-qwen-from-legacy",
        help="Existing qwen retarget files currently include a legacy tool transform; recover Qwen frame by default.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def make_cfg(processed_demo_dir: Path, extrinsics: Path, args: argparse.Namespace):
    output_root, demo_name, demo_num = output_root_from_processed(processed_demo_dir)
    intrinsics = find_intrinsics(output_root, demo_name, demo_num)
    cfg_args = Namespace(
        output_root=output_root,
        demo_name=demo_name,
        demo_num=demo_num,
        input_resolution=args.input_resolution,
        output_resolution=args.output_resolution,
        robot=args.robot,
        gripper=args.gripper,
        bimanual_setup=args.bimanual_setup,
        depth_for_overlay=False,
        depth_occlusion_margin=0.03,
    )
    return build_phantom_cfg(cfg_args, intrinsics, extrinsics), demo_num


def load_raw_frames(processed_demo_dir: Path, size: tuple[int, int] | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(processed_demo_dir / "video_rgb_imgs.mkv"))
    if not cap.isOpened():
        raise FileNotFoundError(processed_demo_dir / "video_rgb_imgs.mkv")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if size is not None and frame.shape[:2] != size:
            frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames


def load_hand_jaw_vectors(processed_demo_dir: Path, setup: str, n_frames: int) -> dict[str, np.ndarray]:
    action_path = processed_demo_dir / "action_processor" / f"actions_left_{setup}.npz"
    hand_dir = processed_demo_dir / "hand_processor"
    with np.load(action_path, allow_pickle=True) as action_data:
        union_indices = np.asarray(action_data["union_indices"], dtype=np.int64)
    vectors: dict[str, np.ndarray] = {}
    for side, path in (("left", hand_dir / "hand_data_left.npz"), ("right", hand_dir / "hand_data_right.npz")):
        with np.load(path) as data:
            kpts = np.asarray(data["kpts_2d"], dtype=np.float64)
            detected = np.asarray(data["hand_detected"], dtype=bool)
        out = np.full((n_frames, 2), np.nan, dtype=np.float64)
        for local_idx, raw_idx in enumerate(union_indices[:n_frames]):
            if raw_idx < 0 or raw_idx >= len(kpts) or not detected[raw_idx]:
                continue
            thumb = kpts[raw_idx, 4]
            virtual_finger = 0.7 * kpts[raw_idx, 8] + 0.3 * kpts[raw_idx, 12]
            vec = thumb - virtual_finger
            norm = np.linalg.norm(vec)
            if np.isfinite(vec).all() and norm > 1e-6:
                out[local_idx] = vec / norm
        vectors[side] = out
    return vectors


def mask_principal_axis(mask: np.ndarray) -> np.ndarray | None:
    mask = np.squeeze(mask).astype(bool)
    if int(mask.sum()) < 8:
        return None
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    axis = vh[0]
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return None
    return axis / norm


def axis_image_error_deg(mask: np.ndarray, target_axis: np.ndarray) -> float:
    axis = mask_principal_axis(mask)
    if axis is None or not np.isfinite(target_axis).all():
        return float("nan")
    target = target_axis / max(np.linalg.norm(target_axis), 1e-8)
    cos = float(np.clip(abs(np.dot(axis, target)), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos)))


def label_cell(cell: np.ndarray, text: str) -> np.ndarray:
    out = cell.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 220), 23), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.squeeze(mask).astype(bool).astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)


def compose(raw: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(np.clip(robot_rgb * 255.0, 0, 255), dtype=np.uint8)
    if raw.shape[:2] != rgb.shape[:2]:
        raw = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = raw.copy()
    mask = np.squeeze(robot_mask).astype(bool)
    if mask.shape != overlay.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[mask] = rgb[mask]
    return overlay


def apply_base_payload(payload: dict) -> None:
    set_base_env(
        np.asarray(payload["base0_offset"], dtype=np.float64),
        np.asarray(payload["base1_offset"], dtype=np.float64),
        float(payload.get("global_yaw_deg", 0.0)),
    )


def main() -> None:
    args = parse_args()
    add_phantom_submodules_to_path()
    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor

    processed_demo_dir = args.processed_demo_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else processed_demo_dir / "gripper_frame_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    traj = load_smoothed(processed_demo_dir, args.bimanual_setup)
    n_frames = min(len(traj["left_pos"]), len(traj["right_pos"]))
    if args.rotation_source == "recover-qwen-from-legacy":
        traj["left_rot"] = recover_qwen_rotations_from_legacy(traj["left_rot"])
        traj["right_rot"] = recover_qwen_rotations_from_legacy(traj["right_rot"])

    frame_ids = [idx for idx in args.frames if 0 <= idx < n_frames]
    if not frame_ids:
        frame_ids = np.linspace(0, n_frames - 1, min(5, n_frames), dtype=int).tolist()
    hand_centers = load_hand_centers(processed_demo_dir, args.bimanual_setup)
    hand_jaw = load_hand_jaw_vectors(processed_demo_dir, args.bimanual_setup, n_frames)
    raw_frames = load_raw_frames(processed_demo_dir)
    base_payload = json.loads(args.base_search_result.resolve().read_text(encoding="utf-8"))
    cfg, demo_num = make_cfg(processed_demo_dir, args.camera_extrinsics.resolve(), args)

    candidates = signed_permutation_candidates()
    apply_base_payload(base_payload)
    processor = None
    rows = []
    candidate_debug: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    try:
        processor = RobotInpaintProcessor(cfg)
        for candidate in candidates:
            processor.twin_robot.reset()
            frame_debug = {}
            pos_errors = []
            right_proj = []
            left_proj = []
            mean_proj = []
            crossing = []
            bottom = []
            robot_area = []
            jaw_image_errors = []
            valid = []
            for local_idx, frame_idx in enumerate(frame_ids):
                right_rot = apply_frame_candidate(traj["right_rot"][frame_idx], candidate)
                left_rot = apply_frame_candidate(traj["left_rot"][frame_idx], candidate)
                target_state = {
                    "pos": [traj["right_pos"][frame_idx], traj["left_pos"][frame_idx]],
                    "ori_xyzw": [
                        Rotation.from_matrix(right_rot).as_quat(scalar_first=False),
                        Rotation.from_matrix(left_rot).as_quat(scalar_first=False),
                    ],
                    "gripper_pos": [traj["right_width"][frame_idx], traj["left_width"][frame_idx]],
                }
                result = processor.twin_robot.move_to_target_state(target_state, init=(local_idx == 0))
                err = max(float(result["left_pos_err"]), float(result["right_pos_err"]))
                visual = visual_metrics(result["robot_mask"], result["gripper_mask"])
                projection = side_projection_metrics(result, hand_centers, int(frame_idx))
                right_jaw_img = axis_image_error_deg(result.get("right_gripper_mask", result["gripper_mask"]), hand_jaw["right"][frame_idx])
                left_jaw_img = axis_image_error_deg(result.get("left_gripper_mask", result["gripper_mask"]), hand_jaw["left"][frame_idx])
                jaw_img = np.nanmean([right_jaw_img, left_jaw_img])
                if not np.isfinite(jaw_img):
                    jaw_img = 90.0
                pos_errors.append(err)
                right_proj.append(projection["right_projection_error_px"])
                left_proj.append(projection["left_projection_error_px"])
                mean_proj.append(projection["mean_projection_error_px"])
                crossing.append(projection["crossing_penalty_px"])
                bottom.append(visual["bottom_area_ratio"])
                robot_area.append(visual["robot_area_ratio"])
                jaw_image_errors.append(float(jaw_img))
                valid.append(err <= args.tracking_threshold)
                frame_debug[int(frame_idx)] = {
                    "overlay": compose(raw_frames[frame_idx], result["rgb_img"], result["robot_mask"] | result["gripper_mask"]),
                    "mask": mask_to_rgb(result["robot_mask"] | result["gripper_mask"]),
                }
            finite_proj = np.asarray(mean_proj, dtype=np.float64)
            mean_projection = float(np.nanmean(finite_proj)) if np.isfinite(finite_proj).any() else 999.0
            mean_crossing = float(np.nanmean(crossing)) if np.isfinite(crossing).any() else 999.0
            mean_jaw_image = float(np.nanmean(jaw_image_errors))
            mean_bottom = float(np.mean(bottom))
            mean_robot = float(np.mean(robot_area))
            mean_pos = float(np.mean(pos_errors))
            valid_ratio = float(np.mean(valid))
            score = (
                mean_pos
                + 0.004 * mean_projection
                + 0.012 * mean_jaw_image
                + 0.010 * mean_crossing
                + 0.80 * mean_bottom
                + 0.10 * mean_robot
            )
            row = {
                "name": candidate.name,
                "matrix": candidate.matrix.tolist(),
                "phantom_jaw_axis_local": candidate.phantom_jaw_axis_local.tolist(),
                "score": float(score),
                "valid_ratio": valid_ratio,
                "mean_tracking_error": mean_pos,
                "max_tracking_error": float(np.max(pos_errors)),
                "mean_projection_error_px": mean_projection,
                "mean_right_projection_error_px": float(np.nanmean(right_proj)),
                "mean_left_projection_error_px": float(np.nanmean(left_proj)),
                "mean_crossing_penalty_px": mean_crossing,
                "mean_jaw_image_error_deg": mean_jaw_image,
                "mean_bottom_area_ratio": mean_bottom,
                "mean_robot_area_ratio": mean_robot,
            }
            rows.append(row)
            candidate_debug[candidate.name] = frame_debug
            print(json.dumps(row), flush=True)
    finally:
        if processor is not None:
            processor.__del__()
        clear_base_env()

    ranked = sorted(rows, key=lambda row: (row["score"], -row["valid_ratio"], row["mean_tracking_error"]))
    payload = {
        "processed_demo_dir": str(processed_demo_dir),
        "base_search_result": str(args.base_search_result.resolve()),
        "rotation_source": args.rotation_source,
        "frames": frame_ids,
        "best": ranked[0],
        "candidates": ranked,
    }
    (output_dir / "frame_sweep_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    montage_rows = []
    montage_rows.append(np.concatenate([label_cell(raw_frames[idx], f"raw f{idx}") for idx in frame_ids], axis=1))
    for rank, row in enumerate(ranked[: args.top_k], start=1):
        debug = candidate_debug[row["name"]]
        label = f"top{rank} score={row['score']:.2f} proj={row['mean_projection_error_px']:.0f} jaw={row['mean_jaw_image_error_deg']:.0f}"
        montage_rows.append(np.concatenate([label_cell(debug[idx]["overlay"], f"{label} f{idx}") for idx in frame_ids], axis=1))
        montage_rows.append(np.concatenate([label_cell(debug[idx]["mask"], f"mask top{rank} f{idx}") for idx in frame_ids], axis=1))
    montage = np.concatenate(montage_rows, axis=0)
    montage_path = output_dir / "frame_sweep_montage.jpg"
    cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"report={output_dir / 'frame_sweep_report.json'}")
    print(f"montage={montage_path}")


if __name__ == "__main__":
    main()
