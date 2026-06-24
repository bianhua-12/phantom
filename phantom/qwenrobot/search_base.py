from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from phantom.qwenrobot.common import (
    DEFAULT_LEFT_EE_BODY,
    DEFAULT_LEFT_EE_OFFSET,
    DEFAULT_LEFT_ROOT_BODY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RIGHT_EE_BODY,
    DEFAULT_RIGHT_EE_OFFSET,
    DEFAULT_RIGHT_ROOT_BODY,
    DEFAULT_URDF,
    read_json,
    representative_bimanual_keyframes,
    safe_id,
    transform_points_from_base,
    transform_rotations_to_base,
    transform_targets_to_base,
    write_json,
)
from phantom.qwenrobot.mujoco_utils import (
    body_id,
    body_point,
    ik_pose_high_precision,
    joint_prefixes_for_body,
    load_model,
    set_parallel_gripper_width,
    style_robot_geoms,
)
from phantom.qwenrobot.visibility_metrics import mask_metrics


def _angle(v: np.ndarray) -> float:
    return float(math.atan2(float(v[1]), float(v[0])))


def _wrap_yaw(yaw: float) -> float:
    return float(((yaw + math.pi) % (2.0 * math.pi)) - math.pi)


def _rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _local_body_pos(mujoco, model, data, body_name: str) -> np.ndarray:
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.xpos[body_id(mujoco, model, body_name)].copy().astype(np.float32)


def _root_target(traj: np.lib.npyio.NpzFile, side: str, anchor: str) -> tuple[np.ndarray, str]:
    preferred = f"{side}_{anchor}_pos"
    if preferred in traj.files:
        return traj[preferred].astype(np.float32), anchor
    for key, name in (
        (f"{side}_shoulder_pos", "shoulder"),
        (f"{side}_arm_pos", "arm"),
        (f"{side}_forearm_pos", "forearm"),
        (f"{side}_hand_pos", "hand"),
    ):
        if key in traj.files:
            return traj[key].astype(np.float32), name
    raise KeyError(f"No {side} root anchor fields found in trajectory npz")


def _candidate_yaws(local_left_root: np.ndarray, local_right_root: np.ndarray, target_left_root: np.ndarray, target_right_root: np.ndarray) -> np.ndarray:
    local_sep = local_left_root[:2] - local_right_root[:2]
    target_sep = np.mean(target_left_root, axis=0)[:2] - np.mean(target_right_root, axis=0)[:2]
    yaws: list[float] = []
    if np.linalg.norm(local_sep) > 1e-5 and np.linalg.norm(target_sep) > 1e-5:
        root_yaw = _wrap_yaw(_angle(target_sep) - _angle(local_sep))
        for deg in (0, -10, 10, -20, 20, -35, 35, -50, 50):
            yaws.append(_wrap_yaw(root_yaw + math.radians(deg)))
    for yaw in (0.0, math.pi, math.pi / 2.0, -math.pi / 2.0):
        yaws.append(_wrap_yaw(yaw))
    return np.unique(np.round(np.asarray(yaws, dtype=np.float32), decimals=5)).astype(np.float32)


def _root_aligned_candidates(
    local_left_root: np.ndarray,
    local_right_root: np.ndarray,
    target_left_root: np.ndarray,
    target_right_root: np.ndarray,
    *,
    grid_radius: float,
    grid_step: float,
    z_offsets: np.ndarray,
) -> np.ndarray:
    target_mean = 0.5 * (np.mean(target_left_root, axis=0) + np.mean(target_right_root, axis=0))
    local_mean = 0.5 * (local_left_root + local_right_root)
    offsets = np.arange(-grid_radius, grid_radius + 1e-6, grid_step, dtype=np.float32)
    rows = []
    for yaw in _candidate_yaws(local_left_root, local_right_root, target_left_root, target_right_root):
        xy = target_mean[:2] - _rot2(float(yaw)) @ local_mean[:2]
        z0 = float(target_mean[2] - local_mean[2])
        for dx in offsets:
            for dy in offsets:
                for dz in z_offsets:
                    rows.append([xy[0] + dx, xy[1] + dy, z0 + float(dz), yaw])
    return np.asarray(rows, dtype=np.float32)


def _centroid_fallback_candidates(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    local_left_root: np.ndarray,
    local_right_root: np.ndarray,
    *,
    grid_radius: float,
    grid_step: float,
    z_offsets: np.ndarray,
) -> np.ndarray:
    centroid = 0.5 * (np.mean(left_xyz, axis=0) + np.mean(right_xyz, axis=0))
    local_root_mean = 0.5 * (local_left_root + local_right_root)
    sep = np.mean(left_xyz, axis=0)[:2] - np.mean(right_xyz, axis=0)[:2]
    yaw0 = float(np.arctan2(-sep[0], sep[1])) if np.linalg.norm(sep) > 1e-6 else 0.0
    xs = np.arange(centroid[0] - grid_radius, centroid[0] + grid_radius + 1e-6, grid_step, dtype=np.float32)
    ys = np.arange(centroid[1] - grid_radius, centroid[1] + grid_radius + 1e-6, grid_step, dtype=np.float32)
    yaws = np.unique(np.round([_wrap_yaw(float(v)) for v in (yaw0, yaw0 + np.pi, yaw0 + 0.5 * np.pi, yaw0 - 0.5 * np.pi, 0.0)], decimals=5)).astype(np.float32)
    rows = []
    for x in xs:
        for y in ys:
            for z in centroid[2] - local_root_mean[2] + z_offsets:
                for yaw in yaws:
                    rows.append([x, y, z, yaw])
    return np.asarray(rows, dtype=np.float32)


def _root_error(base: np.ndarray, local_left_root: np.ndarray, local_right_root: np.ndarray, target_left_root: np.ndarray, target_right_root: np.ndarray) -> float:
    robot_roots = transform_points_from_base(np.stack([local_left_root, local_right_root], axis=0), base)
    targets = np.stack([np.mean(target_left_root, axis=0), np.mean(target_right_root, axis=0)], axis=0)
    return float(np.mean(np.linalg.norm(robot_roots - targets, axis=1)))


def _world_to_camera(points_world: np.ndarray, camera_pos: np.ndarray, camera_rot: np.ndarray) -> np.ndarray:
    return (camera_rot.T @ (points_world - camera_pos).T).T


def _project(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = np.maximum(points_camera[:, 2], 1e-6)
    return np.column_stack(
        [
            intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2],
            intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2],
        ]
    )


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
    return _project(cam, intrinsic)


def _scale_intrinsic(intrinsic: np.ndarray, source_width: int, source_height: int, render_width: int, render_height: int) -> np.ndarray:
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0, :] *= float(render_width) / max(float(source_width), 1.0)
    scaled[1, :] *= float(render_height) / max(float(source_height), 1.0)
    return scaled


def _apply_scene_camera(renderer, pos: np.ndarray, camera_rot: np.ndarray, convention: str) -> None:
    _right, down, forward = _trajectory_camera_axes(camera_rot, convention)
    camera = renderer._scene.camera[0]
    camera.pos[:] = np.asarray(pos, dtype=np.float32)
    camera.forward[:] = forward.astype(np.float32)
    camera.up[:] = (-down).astype(np.float32)


def _robot_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.max(rgb, axis=2) > 12


def _visual_pass_from_metrics(
    metrics: dict[str, float | bool | str | None],
    *,
    min_area: float,
    max_area: float,
    max_center_area: float,
    max_ee_error_px: float,
    max_centroid_error_px: float,
) -> bool:
    if metrics.get("visual_render_error"):
        return False
    mean_area = float(metrics.get("mean_robot_area_ratio") or 0.0)
    max_area_seen = float(metrics.get("max_robot_area_ratio") or 0.0)
    max_center = float(metrics.get("max_robot_center_area_ratio") or 0.0)
    empty_frames = int(metrics.get("empty_robot_frames") or 0)
    ee_error = metrics.get("ee_projection_error_px")
    centroid_error = metrics.get("target_to_robot_centroid_error_px")
    if empty_frames > 0 or mean_area < min_area or max_area_seen > max_area or max_center > max_center_area:
        return False
    if ee_error is not None and float(ee_error) > max_ee_error_px:
        return False
    if centroid_error is not None and float(centroid_error) > max_centroid_error_px:
        return False
    return True


def _visual_veto(
    base: np.ndarray,
    local_left_root: np.ndarray,
    local_right_root: np.ndarray,
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    intrinsic: np.ndarray | None,
    *,
    image_width: int,
    image_height: int,
    center_rect: tuple[float, float, float, float],
    max_center_fraction: float,
    max_bottom_fraction: float,
) -> dict[str, float | bool]:
    if intrinsic is None:
        return {"pass": True, "center_fraction": 0.0, "bottom_fraction": 0.0, "visible_fraction": 0.0}
    base_points_local = np.stack(
        [
            np.zeros(3, dtype=np.float32),
            local_left_root,
            local_right_root,
            0.5 * local_left_root,
            0.5 * local_right_root,
        ],
        axis=0,
    )
    points_world = transform_points_from_base(base_points_local, base)
    center_hits = 0
    bottom_hits = 0
    visible = 0
    total = 0
    x0, x1, y0, y1 = center_rect
    for pos, rot in zip(camera_pos, camera_rot):
        pts_cam = _world_to_camera(points_world, pos.astype(np.float64), rot.astype(np.float64))
        in_front = pts_cam[:, 2] > 0.05
        pixels = _project(pts_cam, intrinsic.astype(np.float64))
        in_image = (
            in_front
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < image_width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < image_height)
        )
        total += int(len(points_world))
        visible += int(np.sum(in_image))
        center_hits += int(
            np.sum(
                in_image
                & (pixels[:, 0] >= x0 * image_width)
                & (pixels[:, 0] <= x1 * image_width)
                & (pixels[:, 1] >= y0 * image_height)
                & (pixels[:, 1] <= y1 * image_height)
            )
        )
        bottom_hits += int(np.sum(in_image & (pixels[:, 1] >= max_bottom_fraction * image_height)))
    center_fraction = center_hits / max(total, 1)
    bottom_fraction = bottom_hits / max(total, 1)
    visible_fraction = visible / max(total, 1)
    return {
        "pass": center_fraction <= max_center_fraction and bottom_fraction <= max_center_fraction,
        "center_fraction": float(center_fraction),
        "bottom_fraction": float(bottom_fraction),
        "visible_fraction": float(visible_fraction),
    }


def _evaluate_candidate(
    mujoco,
    model,
    data,
    *,
    base: np.ndarray,
    keyframes: np.ndarray,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_rot: np.ndarray,
    right_rot: np.ndarray,
    left_ee_id: int,
    right_ee_id: int,
    left_ee_offset: np.ndarray,
    right_ee_offset: np.ndarray,
    left_joint_prefixes: tuple[str, ...] | None,
    right_joint_prefixes: tuple[str, ...] | None,
    ik_tol: float,
    rot_tol: float,
    jaw_axis_tol_deg: float,
    orientation_weight: float,
    jaw_axis_weight: float,
    max_iters: int,
    jaw_axis_local: np.ndarray,
) -> dict[str, float | bool]:
    local_left_targets = transform_targets_to_base(left_xyz[keyframes], base)
    local_left_rots = transform_rotations_to_base(left_rot[keyframes], base)
    local_right_targets = transform_targets_to_base(right_xyz[keyframes], base)
    local_right_rots = transform_rotations_to_base(right_rot[keyframes], base)
    pos_errors = []
    jaw_errors = []
    rot_errors = []
    converged = []
    qpos = None
    for left_target, left_target_rot, right_target, right_target_rot in zip(local_left_targets, local_left_rots, local_right_targets, local_right_rots):
        right = ik_pose_high_precision(
            mujoco,
            model,
            data,
            right_target,
            right_ee_id,
            ee_offset=right_ee_offset,
            target_rot=right_target_rot if orientation_weight > 0.0 else None,
            initial_qpos=qpos,
            tol=ik_tol,
            rot_tol=rot_tol,
            jaw_axis_tol_deg=jaw_axis_tol_deg,
            orientation_weight=orientation_weight,
            jaw_axis_weight=jaw_axis_weight,
            max_iters=max_iters,
            active_joint_prefixes=right_joint_prefixes,
            jaw_axis_local=jaw_axis_local,
        )
        left = ik_pose_high_precision(
            mujoco,
            model,
            data,
            left_target,
            left_ee_id,
            ee_offset=left_ee_offset,
            target_rot=left_target_rot if orientation_weight > 0.0 else None,
            initial_qpos=right.qpos,
            tol=ik_tol,
            rot_tol=rot_tol,
            jaw_axis_tol_deg=jaw_axis_tol_deg,
            orientation_weight=orientation_weight,
            jaw_axis_weight=jaw_axis_weight,
            max_iters=max_iters,
            active_joint_prefixes=left_joint_prefixes,
            jaw_axis_local=jaw_axis_local,
        )
        qpos = left.qpos
        pos_errors.append(max(left.pos_err_m, right.pos_err_m))
        jaw_errors.append(max(left.jaw_axis_err_deg, right.jaw_axis_err_deg))
        rot_errors.append(max(left.rot_err_deg, right.rot_err_deg))
        converged.append(left.converged and right.converged)
    return {
        "valid_ratio": float(np.mean(converged)),
        "mean_pos_error_m": float(np.mean(pos_errors)),
        "max_pos_error_m": float(np.max(pos_errors)),
        "mean_jaw_axis_error_deg": float(np.mean(jaw_errors)),
        "max_jaw_axis_error_deg": float(np.max(jaw_errors)),
        "mean_rotation_error_deg": float(np.mean(rot_errors)),
        "max_rotation_error_deg": float(np.max(rot_errors)),
        "converged": bool(np.all(converged)),
    }


def _render_visual_metrics(
    mujoco,
    model,
    data,
    renderer,
    camera,
    *,
    base: np.ndarray,
    keyframes: np.ndarray,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_rot: np.ndarray,
    right_rot: np.ndarray,
    left_width: np.ndarray,
    right_width: np.ndarray,
    left_camera_xyz: np.ndarray | None,
    right_camera_xyz: np.ndarray | None,
    camera_pos_world: np.ndarray,
    camera_rot_world: np.ndarray,
    intrinsic: np.ndarray | None,
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
    camera_convention: str,
    left_ee_id: int,
    right_ee_id: int,
    left_ee_offset: np.ndarray,
    right_ee_offset: np.ndarray,
    left_joint_prefixes: tuple[str, ...] | None,
    right_joint_prefixes: tuple[str, ...] | None,
    ik_tol: float,
    rot_tol: float,
    jaw_axis_tol_deg: float,
    orientation_weight: float,
    jaw_axis_weight: float,
    max_iters: int,
    jaw_axis_local: np.ndarray,
    min_area: float,
    max_area: float,
    max_center_area: float,
    max_ee_error_px: float,
    max_centroid_error_px: float,
) -> dict[str, float | bool | str | None]:
    if renderer is None or camera is None:
        raise RuntimeError("Visual render scoring requested but renderer is unavailable")
    if len(keyframes) == 0:
        raise ValueError("Visual render scoring requested but no keyframes were selected")

    try:
        local_left_targets = transform_targets_to_base(left_xyz[keyframes], base)
        local_right_targets = transform_targets_to_base(right_xyz[keyframes], base)
        local_left_rots = transform_rotations_to_base(left_rot[keyframes], base)
        local_right_rots = transform_rotations_to_base(right_rot[keyframes], base)
        camera_pos_local = transform_targets_to_base(camera_pos_world[keyframes], base)
        camera_rot_local = transform_rotations_to_base(camera_rot_world[keyframes], base)
        scaled_intrinsic = None
        if intrinsic is not None:
            scaled_intrinsic = _scale_intrinsic(intrinsic, source_width, source_height, render_width, render_height)

        qpos = None
        rows = []
        ee_errors = []
        centroid_errors = []
        for local_i, frame_i in enumerate(keyframes):
            right = ik_pose_high_precision(
                mujoco,
                model,
                data,
                local_right_targets[local_i],
                right_ee_id,
                ee_offset=right_ee_offset,
                target_rot=local_right_rots[local_i] if orientation_weight > 0.0 else None,
                initial_qpos=qpos,
                tol=ik_tol,
                rot_tol=rot_tol,
                jaw_axis_tol_deg=jaw_axis_tol_deg,
                orientation_weight=orientation_weight,
                jaw_axis_weight=jaw_axis_weight,
                max_iters=max_iters,
                active_joint_prefixes=right_joint_prefixes,
                jaw_axis_local=jaw_axis_local,
            )
            left = ik_pose_high_precision(
                mujoco,
                model,
                data,
                local_left_targets[local_i],
                left_ee_id,
                ee_offset=left_ee_offset,
                target_rot=local_left_rots[local_i] if orientation_weight > 0.0 else None,
                initial_qpos=right.qpos,
                tol=ik_tol,
                rot_tol=rot_tol,
                jaw_axis_tol_deg=jaw_axis_tol_deg,
                orientation_weight=orientation_weight,
                jaw_axis_weight=jaw_axis_weight,
                max_iters=max_iters,
                active_joint_prefixes=left_joint_prefixes,
                jaw_axis_local=jaw_axis_local,
            )
            data.qpos[:] = left.qpos
            qpos = left.qpos
            set_parallel_gripper_width(mujoco, model, data, "fr", float(right_width[int(frame_i)]))
            set_parallel_gripper_width(mujoco, model, data, "fl", float(left_width[int(frame_i)]))
            mujoco.mj_forward(model, data)
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=camera)
            _apply_scene_camera(renderer, camera_pos_local[local_i], camera_rot_local[local_i], camera_convention)
            rgb = renderer.render()
            mask = _robot_mask_from_rgb(rgb)
            mask_row = mask_metrics(mask)
            rows.append(mask_row)

            if scaled_intrinsic is not None and left_camera_xyz is not None and right_camera_xyz is not None:
                actual = np.stack(
                    [
                        body_point(data, left_ee_id, left_ee_offset),
                        body_point(data, right_ee_id, right_ee_offset),
                    ],
                    axis=0,
                )
                target_camera = np.stack([left_camera_xyz[int(frame_i)], right_camera_xyz[int(frame_i)]], axis=0)
                actual_px = _project_render_camera(actual, camera_pos_local[local_i], camera_rot_local[local_i], scaled_intrinsic, camera_convention)
                target_px = _project(target_camera, scaled_intrinsic)
                valid = np.isfinite(actual_px).all(axis=1) & np.isfinite(target_px).all(axis=1)
                if np.any(valid):
                    ee_errors.append(float(np.linalg.norm(actual_px[valid] - target_px[valid], axis=1).mean()))
                centroid = mask_row.get("centroid_px")
                if centroid and centroid[0] is not None:
                    target_midpoint = target_px.mean(axis=0)
                    centroid_errors.append(float(np.linalg.norm(target_midpoint - np.asarray(centroid, dtype=np.float64))))

        areas = np.asarray([float(row["area_ratio"]) for row in rows], dtype=np.float64)
        centers = np.asarray([float(row["center_area_ratio"]) for row in rows], dtype=np.float64)
        bottoms = np.asarray([float(row["bottom_area_ratio"]) for row in rows], dtype=np.float64)
        mean_area = float(areas.mean())
        max_area_seen = float(areas.max())
        min_area_seen = float(areas.min())
        max_center_seen = float(centers.max())
        ee_error = float(np.mean(ee_errors)) if ee_errors else None
        centroid_error = float(np.mean(centroid_errors)) if centroid_errors else None
        area_low_penalty = max(0.0, min_area - mean_area)
        area_high_penalty = max(0.0, max_area_seen - max_area)
        center_penalty = max(0.0, max_center_seen - max_center_area)
        ee_penalty = 0.0 if ee_error is None else max(0.0, ee_error - max_ee_error_px) / max(float(render_width), 1.0)
        centroid_penalty = 0.0 if centroid_error is None else max(0.0, centroid_error - max_centroid_error_px) / max(float(render_width), 1.0)
        empty_frames = int(np.sum(areas <= 1e-5))
        visual_score = 8.0 * area_low_penalty + 8.0 * area_high_penalty + 2.0 * center_penalty + ee_penalty + centroid_penalty + 2.0 * empty_frames
        metrics: dict[str, float | bool | str | None] = {
            "visual_render_enabled": True,
            "visual_render_error": None,
            "mean_robot_area_ratio": mean_area,
            "min_robot_area_ratio": min_area_seen,
            "max_robot_area_ratio": max_area_seen,
            "max_robot_center_area_ratio": max_center_seen,
            "max_robot_bottom_area_ratio": float(bottoms.max()),
            "empty_robot_frames": empty_frames,
            "ee_projection_error_px": ee_error,
            "target_to_robot_centroid_error_px": centroid_error,
            "visual_score": float(visual_score),
        }
        metrics["visual_pass"] = _visual_pass_from_metrics(
            metrics,
            min_area=min_area,
            max_area=max_area,
            max_center_area=max_center_area,
            max_ee_error_px=max_ee_error_px,
            max_centroid_error_px=max_centroid_error_px,
        )
        return metrics
    except Exception as exc:
        raise RuntimeError(f"Visual render scoring failed for base {base.tolist()}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Search QwenRobot base placement with synced xyz+yaw transforms and high-precision IK.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--left-ee-body", type=str, default=DEFAULT_LEFT_EE_BODY)
    parser.add_argument("--right-ee-body", type=str, default=DEFAULT_RIGHT_EE_BODY)
    parser.add_argument("--left-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_LEFT_EE_OFFSET))
    parser.add_argument("--right-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_RIGHT_EE_OFFSET))
    parser.add_argument("--left-root-body", type=str, default=DEFAULT_LEFT_ROOT_BODY)
    parser.add_argument("--right-root-body", type=str, default=DEFAULT_RIGHT_ROOT_BODY)
    parser.add_argument("--root-anchor", choices=["shoulder", "arm", "forearm", "hand"], default="shoulder")
    parser.add_argument("--max-root-error", type=float, default=0.45)
    parser.add_argument("--root-grid-radius", type=float, default=0.30)
    parser.add_argument("--root-grid-step", type=float, default=0.10)
    parser.add_argument("--fallback-grid-radius", type=float, default=0.25)
    parser.add_argument("--fallback-grid-step", type=float, default=0.125)
    parser.add_argument("--z-offsets", type=float, nargs="*", default=[-0.15, -0.05, 0.05, 0.15])
    parser.add_argument("--ik-tol", type=float, default=0.005)
    parser.add_argument("--rot-tol", type=float, default=0.35)
    parser.add_argument("--jaw-axis-tol-deg", type=float, default=25.0)
    parser.add_argument("--orientation-weight", type=float, default=0.6)
    parser.add_argument("--jaw-axis-weight", type=float, default=0.25)
    parser.add_argument("--root-weight", type=float, default=0.25)
    parser.add_argument("--visual-weight", type=float, default=0.15)
    parser.add_argument("--max-ik-iters", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--center-rect", type=float, nargs=4, default=[0.15, 0.85, 0.10, 0.90])
    parser.add_argument("--max-center-projection-fraction", type=float, default=0.10)
    parser.add_argument("--bottom-projection-y-frac", type=float, default=0.72)
    parser.add_argument("--jaw-axis-local", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--visual-render-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-render-width", type=int, default=456)
    parser.add_argument("--visual-render-height", type=int, default=256)
    parser.add_argument("--visual-min-area", type=float, default=0.02)
    parser.add_argument("--visual-max-area", type=float, default=0.18)
    parser.add_argument("--visual-max-center-area", type=float, default=0.35)
    parser.add_argument("--visual-max-ee-error-px", type=float, default=80.0)
    parser.add_argument("--visual-max-centroid-error-px", type=float, default=80.0)
    parser.add_argument(
        "--camera-convention",
        choices=["egodex_robot_camera", "mujoco_x_forward", "opencv_z_forward", "neg_z_forward_neg_x_up"],
        default="opencv_z_forward",
    )
    args = parser.parse_args()

    mujoco, model, data = load_model(args.urdf)
    style_robot_geoms(mujoco, model, hide_collision_primitives=True)
    left_ee_id = body_id(mujoco, model, args.left_ee_body)
    right_ee_id = body_id(mujoco, model, args.right_ee_body)
    left_joint_prefixes = joint_prefixes_for_body(args.left_ee_body)
    right_joint_prefixes = joint_prefixes_for_body(args.right_ee_body)
    local_left_root = _local_body_pos(mujoco, model, data, args.left_root_body)
    local_right_root = _local_body_pos(mujoco, model, data, args.right_root_body)
    z_offsets = np.asarray(args.z_offsets, dtype=np.float32)

    rows = []
    manifest = read_json(args.output_dir / "00_manifest.json")
    for row in manifest["episodes"]:
        traj = np.load(row["trajectory_npz"], allow_pickle=True)
        left_xyz = traj["left_ee_pos"].astype(np.float32)
        right_xyz = traj["right_ee_pos"].astype(np.float32)
        left_rot = traj["left_ee_rot"].astype(np.float32)
        right_rot = traj["right_ee_rot"].astype(np.float32)
        keyframes = representative_bimanual_keyframes(left_xyz, right_xyz)
        if len(left_xyz) > 0:
            keyframes = np.unique(np.concatenate([np.asarray([0, len(left_xyz) - 1], dtype=np.int64), keyframes]))
        left_root_target, left_anchor_name = _root_target(traj, "left", args.root_anchor)
        right_root_target, right_anchor_name = _root_target(traj, "right", args.root_anchor)
        root_candidates = _root_aligned_candidates(
            local_left_root,
            local_right_root,
            left_root_target[keyframes],
            right_root_target[keyframes],
            grid_radius=args.root_grid_radius,
            grid_step=args.root_grid_step,
            z_offsets=z_offsets,
        )
        fallback_candidates = _centroid_fallback_candidates(
            left_xyz[keyframes],
            right_xyz[keyframes],
            local_left_root,
            local_right_root,
            grid_radius=args.fallback_grid_radius,
            grid_step=args.fallback_grid_step,
            z_offsets=z_offsets,
        )
        candidates = np.unique(np.round(np.concatenate([root_candidates, fallback_candidates], axis=0), decimals=5), axis=0)
        intrinsic = traj["camera_intrinsic"].astype(np.float32) if "camera_intrinsic" in traj.files else None
        camera_pos = traj["camera_pos"].astype(np.float32) if "camera_pos" in traj.files else np.zeros((len(keyframes), 3), dtype=np.float32)
        camera_rot = traj["camera_rot"].astype(np.float32) if "camera_rot" in traj.files else np.repeat(np.eye(3, dtype=np.float32)[None], len(keyframes), axis=0)
        camera_pos_key = camera_pos[keyframes] if len(camera_pos) >= int(np.max(keyframes)) + 1 else camera_pos
        camera_rot_key = camera_rot[keyframes] if len(camera_rot) >= int(np.max(keyframes)) + 1 else camera_rot
        visual_renderer = None
        visual_camera = None
        if args.visual_render_score:
            try:
                model.vis.global_.offwidth = int(args.visual_render_width)
                model.vis.global_.offheight = int(args.visual_render_height)
                if intrinsic is not None:
                    scaled_fy = float(intrinsic[1, 1]) * float(args.visual_render_height) / max(float(args.image_height), 1.0)
                    model.vis.global_.fovy = math.degrees(2.0 * math.atan(float(args.visual_render_height) / (2.0 * scaled_fy)))
                visual_renderer = mujoco.Renderer(model, height=int(args.visual_render_height), width=int(args.visual_render_width))
                visual_camera = mujoco.MjvCamera()
                visual_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            except Exception as exc:
                raise RuntimeError("Failed to initialize MuJoCo visual renderer") from exc

        prescreen = []
        for base in candidates:
            root_err = _root_error(base, local_left_root, local_right_root, left_root_target[keyframes], right_root_target[keyframes])
            visual = _visual_veto(
                base,
                local_left_root,
                local_right_root,
                camera_pos_key,
                camera_rot_key,
                intrinsic,
                image_width=args.image_width,
                image_height=args.image_height,
                center_rect=tuple(float(v) for v in args.center_rect),
                max_center_fraction=args.max_center_projection_fraction,
                max_bottom_fraction=args.bottom_projection_y_frac,
            )
            if root_err <= args.max_root_error and bool(visual["pass"]):
                prescreen.append((root_err, float(visual["center_fraction"]) + float(visual["bottom_fraction"]), base, visual))
        if not prescreen:
            for base in candidates:
                root_err = _root_error(base, local_left_root, local_right_root, left_root_target[keyframes], right_root_target[keyframes])
                visual = _visual_veto(
                    base,
                    local_left_root,
                    local_right_root,
                    camera_pos_key,
                    camera_rot_key,
                    intrinsic,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    center_rect=tuple(float(v) for v in args.center_rect),
                    max_center_fraction=args.max_center_projection_fraction,
                    max_bottom_fraction=args.bottom_projection_y_frac,
                )
                prescreen.append((root_err, float(visual["center_fraction"]) + float(visual["bottom_fraction"]), base, visual))
        prescreen = sorted(prescreen, key=lambda item: (item[0], item[1]))[: int(args.max_candidates)]

        best = None
        candidate_rows = []
        for cand_i, (root_err, visual_penalty, base, visual) in enumerate(prescreen, start=1):
            print(
                f"evaluate {row['id']} candidate {cand_i}/{len(prescreen)} "
                f"base={np.round(base, 4).tolist()} root_err={root_err:.4f}",
                flush=True,
            )
            ik_metrics = _evaluate_candidate(
                mujoco,
                model,
                data,
                base=base,
                keyframes=keyframes,
                left_xyz=left_xyz,
                right_xyz=right_xyz,
                left_rot=left_rot,
                right_rot=right_rot,
                left_ee_id=left_ee_id,
                right_ee_id=right_ee_id,
                left_ee_offset=np.asarray(args.left_ee_offset, dtype=np.float64),
                right_ee_offset=np.asarray(args.right_ee_offset, dtype=np.float64),
                left_joint_prefixes=left_joint_prefixes,
                right_joint_prefixes=right_joint_prefixes,
                ik_tol=args.ik_tol,
                rot_tol=args.rot_tol,
                jaw_axis_tol_deg=args.jaw_axis_tol_deg,
                orientation_weight=args.orientation_weight,
                jaw_axis_weight=args.jaw_axis_weight,
                max_iters=max(500, int(args.max_ik_iters)),
                jaw_axis_local=np.asarray(args.jaw_axis_local, dtype=np.float64),
            )
            visual_metrics = _render_visual_metrics(
                mujoco,
                model,
                data,
                visual_renderer,
                visual_camera,
                base=base,
                keyframes=keyframes,
                left_xyz=left_xyz,
                right_xyz=right_xyz,
                left_rot=left_rot,
                right_rot=right_rot,
                left_width=traj["left_gripper_width"].astype(np.float32),
                right_width=traj["right_gripper_width"].astype(np.float32),
                left_camera_xyz=traj["left_ee_pos_camera"].astype(np.float32) if "left_ee_pos_camera" in traj.files else None,
                right_camera_xyz=traj["right_ee_pos_camera"].astype(np.float32) if "right_ee_pos_camera" in traj.files else None,
                camera_pos_world=traj["camera_pos"].astype(np.float32) if "camera_pos" in traj.files else np.zeros((len(left_xyz), 3), dtype=np.float32),
                camera_rot_world=traj["camera_rot"].astype(np.float32) if "camera_rot" in traj.files else np.repeat(np.eye(3, dtype=np.float32)[None], len(left_xyz), axis=0),
                intrinsic=intrinsic,
                source_width=args.image_width,
                source_height=args.image_height,
                render_width=args.visual_render_width,
                render_height=args.visual_render_height,
                camera_convention=args.camera_convention,
                left_ee_id=left_ee_id,
                right_ee_id=right_ee_id,
                left_ee_offset=np.asarray(args.left_ee_offset, dtype=np.float64),
                right_ee_offset=np.asarray(args.right_ee_offset, dtype=np.float64),
                left_joint_prefixes=left_joint_prefixes,
                right_joint_prefixes=right_joint_prefixes,
                ik_tol=args.ik_tol,
                rot_tol=args.rot_tol,
                jaw_axis_tol_deg=args.jaw_axis_tol_deg,
                orientation_weight=args.orientation_weight,
                jaw_axis_weight=args.jaw_axis_weight,
                max_iters=max(80, int(args.max_ik_iters)),
                jaw_axis_local=np.asarray(args.jaw_axis_local, dtype=np.float64),
                min_area=args.visual_min_area,
                max_area=args.visual_max_area,
                max_center_area=args.visual_max_center_area,
                max_ee_error_px=args.visual_max_ee_error_px,
                max_centroid_error_px=args.visual_max_centroid_error_px,
            )
            objective = (
                float(ik_metrics["mean_pos_error_m"])
                + 0.0025 * float(ik_metrics["mean_jaw_axis_error_deg"])
                + 0.0015 * float(ik_metrics["mean_rotation_error_deg"])
                + args.root_weight * root_err
                + args.visual_weight * visual_penalty
                + 0.05 * float(visual_metrics.get("visual_score") or 0.0)
            )
            candidate = {
                "base_xyz_yaw": base.astype(np.float32),
                "base_xyyaw": np.asarray([base[0], base[1], base[3]], dtype=np.float32),
                "root_error": float(root_err),
                "root_center_fraction": float(visual["center_fraction"]),
                "visual_center_fraction": float(visual["center_fraction"]),
                "visual_bottom_fraction": float(visual["bottom_fraction"]),
                "visual_visible_fraction": float(visual["visible_fraction"]),
                "visual_base_veto_pass": bool(visual["pass"]),
                "objective": float(objective),
                **ik_metrics,
                **visual_metrics,
            }
            candidate_rows.append(candidate)
            visual_sort_enabled = bool(candidate.get("visual_render_enabled")) and candidate.get("visual_render_error") in (None, "")
            sort_key = (
                float(candidate["valid_ratio"]),
                1.0 if (not visual_sort_enabled or bool(candidate.get("visual_pass"))) else 0.0,
                -float(candidate.get("visual_score") or 0.0),
                -float(candidate["mean_pos_error_m"]),
                -float(candidate["mean_jaw_axis_error_deg"]),
                -float(candidate["mean_rotation_error_deg"]),
                -float(candidate["visual_center_fraction"]),
                -float(candidate["root_error"]),
            )
            if best is None or sort_key > best["_sort_key"]:
                best = {**candidate, "_sort_key": sort_key}
        if best is None:
            raise RuntimeError(f"No base candidates evaluated for {row['id']}")
        if visual_renderer is not None:
            visual_renderer.close()

        out_npz = args.output_dir / "03_base_search" / f"{safe_id(row['id'])}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        candidate_metric_names = [
            "x",
            "y",
            "z",
            "yaw",
            "valid_ratio",
            "mean_pos_error_m",
            "max_pos_error_m",
            "mean_jaw_axis_error_deg",
            "max_jaw_axis_error_deg",
            "mean_rotation_error_deg",
            "max_rotation_error_deg",
            "root_error",
            "visual_center_fraction",
            "visual_bottom_fraction",
            "visual_visible_fraction",
            "root_center_fraction",
            "mean_robot_area_ratio",
            "min_robot_area_ratio",
            "max_robot_area_ratio",
            "max_robot_center_area_ratio",
            "max_robot_bottom_area_ratio",
            "empty_robot_frames",
            "ee_projection_error_px",
            "target_to_robot_centroid_error_px",
            "visual_score",
            "objective",
        ]
        candidate_metrics = np.asarray(
            [
                [
                    *np.asarray(c["base_xyz_yaw"], dtype=np.float32).tolist(),
                    float(c["valid_ratio"]),
                    float(c["mean_pos_error_m"]),
                    float(c["max_pos_error_m"]),
                    float(c["mean_jaw_axis_error_deg"]),
                    float(c["max_jaw_axis_error_deg"]),
                    float(c["mean_rotation_error_deg"]),
                    float(c["max_rotation_error_deg"]),
                    float(c["root_error"]),
                    float(c["visual_center_fraction"]),
                    float(c["visual_bottom_fraction"]),
                    float(c["visual_visible_fraction"]),
                    float(c.get("root_center_fraction") or 0.0),
                    float(c.get("mean_robot_area_ratio") or 0.0),
                    float(c.get("min_robot_area_ratio") or 0.0),
                    float(c.get("max_robot_area_ratio") or 0.0),
                    float(c.get("max_robot_center_area_ratio") or 0.0),
                    float(c.get("max_robot_bottom_area_ratio") or 0.0),
                    float(c.get("empty_robot_frames") or 0.0),
                    float(c.get("ee_projection_error_px") or 0.0),
                    float(c.get("target_to_robot_centroid_error_px") or 0.0),
                    float(c.get("visual_score") or 0.0),
                    float(c["objective"]),
                ]
                for c in candidate_rows
            ],
            dtype=np.float32,
        )
        best_metrics = {k: v for k, v in best.items() if k != "_sort_key"}
        np.savez_compressed(
            out_npz,
            base_xyz_yaw=np.asarray(best["base_xyz_yaw"], dtype=np.float32),
            base_xyyaw=np.asarray(best["base_xyyaw"], dtype=np.float32),
            keyframes=keyframes,
            candidate_metric_names=np.asarray(candidate_metric_names),
            candidate_metrics=candidate_metrics,
            best_metrics=json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_metrics.items()}),
            local_left_root=local_left_root,
            local_right_root=local_right_root,
            left_root_target=left_root_target[keyframes],
            right_root_target=right_root_target[keyframes],
        )
        rows.append(
            {
                "id": row["id"],
                "base_npz": out_npz,
                "base_xyz_yaw": best["base_xyz_yaw"],
                "base_xyyaw": best["base_xyyaw"],
                "keyframe_feasible_ratio": best["valid_ratio"],
                "mean_pos_error_m": best["mean_pos_error_m"],
                "max_pos_error_m": best["max_pos_error_m"],
                "mean_jaw_axis_error_deg": best["mean_jaw_axis_error_deg"],
                "max_jaw_axis_error_deg": best["max_jaw_axis_error_deg"],
                "mean_rotation_error_deg": best["mean_rotation_error_deg"],
                "max_rotation_error_deg": best["max_rotation_error_deg"],
                "root_error": best["root_error"],
                "visual_base_veto_pass": best["visual_base_veto_pass"],
                "visual_center_fraction": best["visual_center_fraction"],
                "visual_bottom_fraction": best["visual_bottom_fraction"],
                "root_center_fraction": best.get("root_center_fraction"),
                "visual_render_enabled": best.get("visual_render_enabled"),
                "visual_render_error": best.get("visual_render_error"),
                "visual_pass": best.get("visual_pass"),
                "mean_robot_area_ratio": best.get("mean_robot_area_ratio"),
                "min_robot_area_ratio": best.get("min_robot_area_ratio"),
                "max_robot_area_ratio": best.get("max_robot_area_ratio"),
                "max_robot_center_area_ratio": best.get("max_robot_center_area_ratio"),
                "max_robot_bottom_area_ratio": best.get("max_robot_bottom_area_ratio"),
                "empty_robot_frames": best.get("empty_robot_frames"),
                "ee_projection_error_px": best.get("ee_projection_error_px"),
                "target_to_robot_centroid_error_px": best.get("target_to_robot_centroid_error_px"),
                "visual_score": best.get("visual_score"),
                "objective": best["objective"],
                "urdf": args.urdf,
                "left_ee_body": args.left_ee_body,
                "left_ee_offset": args.left_ee_offset,
                "right_ee_body": args.right_ee_body,
                "right_ee_offset": args.right_ee_offset,
                "left_root_body": args.left_root_body,
                "right_root_body": args.right_root_body,
                "root_anchor": args.root_anchor,
                "left_anchor_used": left_anchor_name,
                "right_anchor_used": right_anchor_name,
                "max_root_error": args.max_root_error,
                "root_weight": args.root_weight,
                "visual_weight": args.visual_weight,
                "visual_render_width": args.visual_render_width,
                "visual_render_height": args.visual_render_height,
                "visual_min_area": args.visual_min_area,
                "visual_max_area": args.visual_max_area,
                "visual_max_center_area": args.visual_max_center_area,
                "visual_max_ee_error_px": args.visual_max_ee_error_px,
                "visual_max_centroid_error_px": args.visual_max_centroid_error_px,
                "camera_convention": args.camera_convention,
                "num_candidates": int(len(candidates)),
                "num_prescreened_candidates": int(len(prescreen)),
            }
        )
    write_json(args.output_dir / "03_manifest.json", {"stage": "qwenrobot_mujoco_base_search_xyz_yaw_high_precision", "episodes": rows})
    print(args.output_dir / "03_manifest.json")


if __name__ == "__main__":
    main()
