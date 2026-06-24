from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from phantom.qwenrobot.common import (
    DEFAULT_LEFT_EE_BODY,
    DEFAULT_LEFT_EE_OFFSET,
    DEFAULT_RIGHT_EE_BODY,
    DEFAULT_RIGHT_EE_OFFSET,
    DEFAULT_URDF,
    read_json,
    safe_id,
    transform_rotations_to_base,
    transform_targets_to_base,
    write_json,
)
from phantom.qwenrobot.visibility_metrics import (
    chain_report,
    classify_failure,
    finite_mean_distance,
    json_sanitize,
    mask_metrics,
    projection_from_camera_points,
    summarize_frame_metrics,
)


def _robot_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    mask = np.max(rgb, axis=2) > 12
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def _overlay(raw: np.ndarray, robot: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mask = _robot_mask_from_rgb(robot)
    out = raw.copy()
    if alpha >= 0.999:
        out[mask] = robot[mask]
    else:
        out[mask] = (alpha * robot[mask].astype(np.float32) + (1.0 - alpha) * raw[mask].astype(np.float32)).astype(np.uint8)
    return out, mask


def _read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_optional_frame(path: Path | None, index: int, fallback: np.ndarray) -> np.ndarray:
    if path is None:
        return fallback.copy()
    if not path.exists():
        raise FileNotFoundError(f"Optional video was provided but does not exist: {path}")
    return _read_frame(path, index)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], max(160, 9 * len(text))), 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _mask_panel(mask: np.ndarray, *, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[np.asarray(mask).astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _draw_points(image: np.ndarray, points: dict[str, tuple[float, float] | None]) -> np.ndarray:
    out = image.copy()
    colors = {
        "left_target": (0, 255, 255),
        "right_target": (255, 0, 255),
        "left_actual": (255, 255, 0),
        "right_actual": (255, 120, 255),
        "left_root": (0, 200, 0),
        "right_root": (255, 120, 0),
    }
    for name, point in points.items():
        if point is None:
            continue
        x, y = point
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, 7, colors.get(name, (255, 255, 255)), 2, cv2.LINE_AA)
        cv2.putText(out, name.replace("_", " "), (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.get(name, (255, 255, 255)), 1, cv2.LINE_AA)
    return out


def _project_targets(traj: np.lib.npyio.NpzFile, frame_i: int) -> dict[str, tuple[float, float] | None]:
    if "camera_intrinsic" not in traj.files:
        return {"left_target": None, "right_target": None}
    intrinsic = traj["camera_intrinsic"].astype(np.float64)
    out: dict[str, tuple[float, float] | None] = {}
    for side in ("left", "right"):
        key = f"{side}_ee_pos_camera"
        if key not in traj.files or frame_i >= len(traj[key]):
            out[f"{side}_target"] = None
            continue
        px = projection_from_camera_points(traj[key][frame_i : frame_i + 1], intrinsic)[0]
        out[f"{side}_target"] = (float(px[0]), float(px[1]))
    return out


def _project_roots(qpos: np.lib.npyio.NpzFile, traj: np.lib.npyio.NpzFile, frame_i: int) -> dict[str, tuple[float, float] | None]:
    if "root_positions" not in qpos.files or "camera_intrinsic" not in traj.files or "camera_pos" not in traj.files or "camera_rot" not in traj.files:
        return {"left_root": None, "right_root": None}
    if frame_i >= len(qpos["root_positions"]) or frame_i >= len(traj["camera_pos"]):
        return {"left_root": None, "right_root": None}
    roots_world = np.asarray(qpos["root_positions"][frame_i], dtype=np.float64).reshape(-1, 3)
    cam_pos = np.asarray(traj["camera_pos"][frame_i], dtype=np.float64)
    cam_rot = np.asarray(traj["camera_rot"][frame_i], dtype=np.float64)
    roots_camera = (cam_rot.T @ (roots_world - cam_pos).T).T
    px = projection_from_camera_points(roots_camera, traj["camera_intrinsic"].astype(np.float64))
    return {
        "left_root": (float(px[0, 0]), float(px[0, 1])) if len(px) > 0 else None,
        "right_root": (float(px[1, 0]), float(px[1, 1])) if len(px) > 1 else None,
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _trajectory_camera_axes(camera_rot: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = np.asarray(camera_rot, dtype=np.float64).reshape(3, 3)
    if convention == "egodex_robot_camera":
        forward = -rot[:, 0]
        up = -rot[:, 2]
    elif convention == "opencv_z_forward":
        forward = rot[:, 2]
        up = -rot[:, 1]
    elif convention == "mujoco_x_forward":
        forward = rot[:, 0]
        up = rot[:, 1]
    elif convention == "neg_z_forward_neg_x_up":
        forward = -rot[:, 2]
        up = -rot[:, 0]
    else:
        raise ValueError(f"Unknown camera convention: {convention}")
    forward = _normalize(forward)
    up = _normalize(up - forward * float(np.dot(up, forward)))
    right = _normalize(np.cross(forward, up))
    down = -up
    return right, down, forward


def _project_render_camera(points: np.ndarray, camera_pos: np.ndarray, camera_rot: np.ndarray, intrinsic: np.ndarray, convention: str) -> np.ndarray:
    right, down, forward = _trajectory_camera_axes(camera_rot, convention)
    basis = np.column_stack([right, down, forward])
    cam = (basis.T @ (np.asarray(points, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)).T).T
    return projection_from_camera_points(cam, intrinsic)


def _build_actual_ee_projector(
    qpos: np.lib.npyio.NpzFile | None,
    traj: np.lib.npyio.NpzFile,
    *,
    urdf: Path,
    left_ee_body: str,
    right_ee_body: str,
    left_ee_offset: np.ndarray,
    right_ee_offset: np.ndarray,
    camera_convention: str,
):
    if qpos is None or "qpos" not in qpos.files or "camera_intrinsic" not in traj.files:
        return None, "missing_qpos_or_intrinsic"
    try:
        from phantom.qwenrobot.mujoco_utils import body_id, body_point, load_model
    except Exception as exc:
        return None, f"mujoco_unavailable:{exc!r}"
    try:
        mujoco, model, data = load_model(urdf)
        left_id = body_id(mujoco, model, left_ee_body)
        right_id = body_id(mujoco, model, right_ee_body)
    except Exception as exc:
        return None, f"load_model_failed:{exc!r}"

    placement = str(qpos["placement_mode"]) if "placement_mode" in qpos.files else "world_base"
    base = np.asarray(qpos["base_xyz_yaw"], dtype=np.float32) if "base_xyz_yaw" in qpos.files else np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    intrinsic = traj["camera_intrinsic"].astype(np.float64)

    def project(frame_i: int) -> dict[str, tuple[float, float] | None]:
        if frame_i >= len(qpos["qpos"]):
            return {"left_actual": None, "right_actual": None}
        data.qpos[:] = qpos["qpos"][frame_i]
        mujoco.mj_forward(model, data)
        actual = np.stack(
            [
                body_point(data, left_id, left_ee_offset),
                body_point(data, right_id, right_ee_offset),
            ],
            axis=0,
        )
        if placement == "world_base":
            camera_pos = transform_targets_to_base(traj["camera_pos"][frame_i : frame_i + 1].astype(np.float32), base)[0]
            camera_rot = transform_rotations_to_base(traj["camera_rot"][frame_i : frame_i + 1].astype(np.float32), base)[0]
            px = _project_render_camera(actual, camera_pos, camera_rot, intrinsic, camera_convention)
        else:
            px = projection_from_camera_points(actual, intrinsic)
        return {
            "left_actual": (float(px[0, 0]), float(px[0, 1])),
            "right_actual": (float(px[1, 0]), float(px[1, 1])),
        }

    return project, None


def _mean_point(points: list[tuple[float, float] | None]) -> tuple[float, float] | None:
    vals = np.asarray([p for p in points if p is not None and np.isfinite(p).all()], dtype=np.float64)
    if vals.size == 0:
        return None
    mean = vals.reshape(-1, 2).mean(axis=0)
    return (float(mean[0]), float(mean[1]))


def _point_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if not (np.isfinite(arr_a).all() and np.isfinite(arr_b).all()):
        return None
    return float(np.linalg.norm(arr_a - arr_b))


def _frame_indices(requested: list[int], n_frames: int) -> list[int]:
    out = []
    for idx in requested:
        if 0 <= idx < n_frames:
            out.append(int(idx))
    if out:
        return sorted(dict.fromkeys(out))
    defaults = np.linspace(0, max(0, n_frames - 1), min(5, max(1, n_frames)), dtype=int).tolist()
    return sorted(dict.fromkeys(defaults))


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def diagnose_episode(
    output_dir: Path,
    episode: dict[str, Any],
    render_row: dict[str, Any] | None,
    requested_frames: list[int],
    out_dir: Path,
    *,
    urdf: Path,
    left_ee_body: str,
    right_ee_body: str,
    left_ee_offset: np.ndarray,
    right_ee_offset: np.ndarray,
    camera_convention: str,
) -> dict[str, Any]:
    sid = safe_id(episode["id"])
    traj = np.load(episode["trajectory_npz"], allow_pickle=True)
    qpos_path = Path(render_row["qpos_npz"]) if render_row and "qpos_npz" in render_row else output_dir / "04_ik_qpos" / f"{sid}.npz"
    qpos = np.load(qpos_path, allow_pickle=True) if qpos_path.exists() else None
    raw_video = Path(episode["sampled_video"])
    robot_video = Path(render_row["robot_video"]) if render_row and "robot_video" in render_row else output_dir / "04_robot_rgb" / f"{sid}.mp4"
    overlay_video = Path(render_row["raw_robot_overlay_video"]) if render_row and "raw_robot_overlay_video" in render_row else output_dir / "04_raw_robot_overlay" / f"{sid}.mp4"
    n_frames = min(_video_frame_count(raw_video), _video_frame_count(robot_video) or 10**9, len(traj["left_ee_pos"]))
    frames = _frame_indices(requested_frames, n_frames)
    episode_dir = out_dir / sid
    episode_dir.mkdir(parents=True, exist_ok=True)
    actual_projector, actual_projector_error = _build_actual_ee_projector(
        qpos,
        traj,
        urdf=urdf,
        left_ee_body=left_ee_body,
        right_ee_body=right_ee_body,
        left_ee_offset=left_ee_offset,
        right_ee_offset=right_ee_offset,
        camera_convention=camera_convention,
    )

    rows: list[dict[str, Any]] = []
    montage_cols: list[np.ndarray] = []
    for frame_i in frames:
        raw = _read_frame(raw_video, frame_i)
        robot = _read_optional_frame(robot_video, frame_i, np.zeros_like(raw))
        overlay_existing = _read_optional_frame(overlay_video, frame_i, raw)
        if robot.shape[:2] != raw.shape[:2]:
            robot = cv2.resize(robot, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = _robot_mask_from_rgb(robot)
        overlay_rebuilt, visible_mask = _overlay(raw, robot, 1.0)
        occluded_mask = mask & ~visible_mask
        target_px = _project_targets(traj, frame_i)
        actual_px = actual_projector(frame_i) if actual_projector is not None else {"left_actual": None, "right_actual": None}
        root_px = _project_roots(qpos, traj, frame_i) if qpos is not None else {"left_root": None, "right_root": None}
        target_panel = _draw_points(raw, {**target_px, **actual_px, **root_px})
        mm = mask_metrics(mask)
        vm = mask_metrics(visible_mask)
        om = mask_metrics(occluded_mask)
        target_midpoint = _mean_point([target_px["left_target"], target_px["right_target"]])
        robot_centroid = None
        if mm["centroid_px"][0] is not None:
            robot_centroid = (float(mm["centroid_px"][0]), float(mm["centroid_px"][1]))
        row = {
            "frame": int(frame_i),
            "robot_area_ratio": mm["area_ratio"],
            "visible_area_ratio": vm["area_ratio"],
            "occluded_area_ratio": om["area_ratio"],
            "robot_center_area_ratio": mm["center_area_ratio"],
            "robot_bottom_area_ratio": mm["bottom_area_ratio"],
            "robot_bbox_px": mm["bbox_px"],
            "robot_centroid_px": mm["centroid_px"],
            "left_target_px": target_px["left_target"],
            "right_target_px": target_px["right_target"],
            "left_actual_ee_px": actual_px["left_actual"],
            "right_actual_ee_px": actual_px["right_actual"],
            "left_root_px": root_px["left_root"],
            "right_root_px": root_px["right_root"],
            "root_target_projection_error_px": finite_mean_distance(
                np.asarray([target_px["left_target"], target_px["right_target"]], dtype=np.float64),
                np.asarray([root_px["left_root"], root_px["right_root"]], dtype=np.float64),
            ),
            "target_midpoint_px": target_midpoint,
            "target_to_robot_centroid_error_px": _point_distance(target_midpoint, robot_centroid),
            "ee_projection_error_px": finite_mean_distance(
                np.asarray([target_px["left_target"], target_px["right_target"]], dtype=np.float64),
                np.asarray([actual_px["left_actual"], actual_px["right_actual"]], dtype=np.float64),
            ),
            "pos_error_m": None,
            "jaw_axis_error_deg": None,
            "rotation_error_deg": None,
            "converged": None,
        }
        if qpos is not None and "pos_error_m" in qpos.files and frame_i < len(qpos["pos_error_m"]):
            row["pos_error_m"] = float(np.max(qpos["pos_error_m"][frame_i]))
            row["jaw_axis_error_deg"] = float(np.max(qpos["jaw_axis_error_deg"][frame_i]))
            row["rotation_error_deg"] = float(np.max(qpos["rotation_error_deg"][frame_i]))
            row["converged"] = bool(np.all(qpos["converged"][frame_i]))
        rows.append(row)

        cv2.imwrite(str(episode_dir / f"{frame_i:06d}_raw_targets.png"), cv2.cvtColor(target_panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(episode_dir / f"{frame_i:06d}_robot.png"), cv2.cvtColor(robot, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(episode_dir / f"{frame_i:06d}_overlay_existing.png"), cv2.cvtColor(overlay_existing, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(episode_dir / f"{frame_i:06d}_overlay_rebuilt.png"), cv2.cvtColor(overlay_rebuilt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(episode_dir / f"{frame_i:06d}_robot_mask.png"), _mask_panel(mask))

        col = np.concatenate(
            [
                _label(target_panel, f"raw targets idx{frame_i}"),
                _label(robot, f"robot idx{frame_i}"),
                _label(overlay_rebuilt, f"rebuilt overlay idx{frame_i}"),
                _label(_mask_panel(mask), f"robot mask idx{frame_i}"),
                _label(_mask_panel(visible_mask, color=(0, 255, 0)) + _mask_panel(occluded_mask, color=(255, 0, 0)), f"visible/occ idx{frame_i}"),
            ],
            axis=0,
        )
        montage_cols.append(col)

    montage = np.concatenate(montage_cols, axis=1)
    montage_path = episode_dir / "visibility_montage.jpg"
    cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    csv_path = episode_dir / "visibility_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame"])
        writer.writeheader()
        writer.writerows([{k: json.dumps(v) if isinstance(v, (list, tuple)) else v for k, v in row.items()} for row in rows])

    summary = summarize_frame_metrics(rows)
    if render_row:
        for key in ("mean_pos_error_m", "mean_jaw_axis_error_deg", "converged_ratio", "camera_convention", "depth_aware_overlay", "mean_visible_robot_area", "mean_occluded_robot_area"):
            if key in render_row:
                summary[f"render_{key}"] = render_row[key]
    summary["failure_classification"] = classify_failure(summary)
    summary["chain_report"] = chain_report(summary)
    summary["actual_ee_projector_error"] = actual_projector_error
    summary["episode_id"] = episode["id"]
    summary["frames"] = frames
    summary["montage"] = str(montage_path)
    summary["csv"] = str(csv_path)
    summary["raw_video"] = str(raw_video)
    summary["robot_video"] = str(robot_video)
    summary["overlay_video"] = str(overlay_video)
    write_json(episode_dir / "visibility_summary.json", json_sanitize(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose single-demo QwenRobot visibility, projection, and overlay geometry from existing artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwenrobot_egodex_episode0_shoulder"))
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 30, 60, 90, 119])
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--left-ee-body", type=str, default=DEFAULT_LEFT_EE_BODY)
    parser.add_argument("--right-ee-body", type=str, default=DEFAULT_RIGHT_EE_BODY)
    parser.add_argument("--left-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_LEFT_EE_OFFSET))
    parser.add_argument("--right-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_RIGHT_EE_OFFSET))
    parser.add_argument(
        "--camera-convention",
        choices=["egodex_robot_camera", "mujoco_x_forward", "opencv_z_forward", "neg_z_forward_neg_x_up"],
        default=None,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    out_dir = args.diagnostics_dir.resolve() if args.diagnostics_dir is not None else output_dir / "09_visibility_diagnostics"
    manifest = read_json(output_dir / "00_manifest.json")
    render_manifest_path = output_dir / "04_manifest.json"
    render_rows = {}
    if render_manifest_path.exists():
        render_rows = {row["id"]: row for row in read_json(render_manifest_path)["episodes"]}
    episodes = manifest["episodes"]
    if args.episode_id is not None:
        episodes = [row for row in episodes if row["id"] == args.episode_id]
    if not episodes:
        raise ValueError(f"No matching episodes in {output_dir / '00_manifest.json'}")
    summaries = []
    for row in episodes:
        render_row = render_rows.get(row["id"])
        camera_convention = args.camera_convention or (render_row or {}).get("camera_convention") or "egodex_robot_camera"
        summaries.append(
            diagnose_episode(
                output_dir,
                row,
                render_row,
                args.frames,
                out_dir,
                urdf=args.urdf,
                left_ee_body=args.left_ee_body,
                right_ee_body=args.right_ee_body,
                left_ee_offset=np.asarray(args.left_ee_offset, dtype=np.float64),
                right_ee_offset=np.asarray(args.right_ee_offset, dtype=np.float64),
                camera_convention=camera_convention,
            )
        )
    top_summary = {"output_dir": str(output_dir), "diagnostics_dir": str(out_dir), "episodes": summaries}
    write_json(out_dir / "visibility_manifest.json", json_sanitize(top_summary))
    print(out_dir / "visibility_manifest.json")


if __name__ == "__main__":
    main()
