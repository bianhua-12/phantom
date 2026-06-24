from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_CENTER_RECT = (0.15, 0.85, 0.10, 0.90)


def mask_metrics(
    mask: np.ndarray,
    *,
    center_rect: tuple[float, float, float, float] = DEFAULT_CENTER_RECT,
    bottom_y_frac: float = 0.67,
) -> dict[str, Any]:
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_bool.shape}")
    height, width = mask_bool.shape
    area = float(mask_bool.mean())
    y_idx, x_idx = np.where(mask_bool)
    if len(x_idx) == 0:
        bbox = [None, None, None, None]
        bbox_norm = [None, None, None, None]
        centroid = [None, None]
        centroid_norm = [None, None]
    else:
        x0, x1 = int(x_idx.min()), int(x_idx.max())
        y0, y1 = int(y_idx.min()), int(y_idx.max())
        bbox = [x0, y0, x1, y1]
        bbox_norm = [x0 / width, y0 / height, x1 / width, y1 / height]
        centroid = [float(x_idx.mean()), float(y_idx.mean())]
        centroid_norm = [centroid[0] / width, centroid[1] / height]

    x0r, x1r, y0r, y1r = center_rect
    center = mask_bool[
        int(round(y0r * height)) : int(round(y1r * height)),
        int(round(x0r * width)) : int(round(x1r * width)),
    ]
    bottom = mask_bool[int(round(bottom_y_frac * height)) :, :]
    return {
        "area_ratio": area,
        "bbox_px": bbox,
        "bbox_norm": bbox_norm,
        "centroid_px": centroid,
        "centroid_norm": centroid_norm,
        "center_area_ratio": float(center.mean()) if center.size else 0.0,
        "bottom_area_ratio": float(bottom.mean()) if bottom.size else 0.0,
        "nonzero": bool(len(x_idx) > 0),
        "width": int(width),
        "height": int(height),
    }


def summarize_frame_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "frames": 0,
            "mean_robot_area_ratio": None,
            "min_robot_area_ratio": None,
            "max_robot_area_ratio": None,
            "empty_robot_frames": 0,
            "max_center_area_ratio": None,
            "max_bottom_area_ratio": None,
        }
    robot_area = np.asarray([float(row.get("robot_area_ratio", 0.0)) for row in rows], dtype=np.float64)
    center = np.asarray([float(row.get("robot_center_area_ratio", 0.0)) for row in rows], dtype=np.float64)
    bottom = np.asarray([float(row.get("robot_bottom_area_ratio", 0.0)) for row in rows], dtype=np.float64)
    visible = np.asarray([float(row.get("visible_area_ratio", row.get("robot_area_ratio", 0.0))) for row in rows], dtype=np.float64)
    occluded = np.asarray([float(row.get("occluded_area_ratio", 0.0)) for row in rows], dtype=np.float64)
    ee_errors = [float(row["ee_projection_error_px"]) for row in rows if row.get("ee_projection_error_px") is not None]
    centroid_errors = [
        float(row["target_to_robot_centroid_error_px"])
        for row in rows
        if row.get("target_to_robot_centroid_error_px") is not None
    ]
    root_errors = [
        float(row["root_target_projection_error_px"])
        for row in rows
        if row.get("root_target_projection_error_px") is not None
    ]
    return {
        "frames": int(len(rows)),
        "mean_robot_area_ratio": float(robot_area.mean()),
        "min_robot_area_ratio": float(robot_area.min()),
        "max_robot_area_ratio": float(robot_area.max()),
        "empty_robot_frames": int(np.sum(robot_area <= 1e-5)),
        "mean_visible_area_ratio": float(visible.mean()),
        "mean_occluded_area_ratio": float(occluded.mean()),
        "max_center_area_ratio": float(center.max()),
        "max_bottom_area_ratio": float(bottom.max()),
        "mean_ee_projection_error_px": float(np.mean(ee_errors)) if ee_errors else None,
        "max_ee_projection_error_px": float(np.max(ee_errors)) if ee_errors else None,
        "mean_target_to_robot_centroid_error_px": float(np.mean(centroid_errors)) if centroid_errors else None,
        "max_target_to_robot_centroid_error_px": float(np.max(centroid_errors)) if centroid_errors else None,
        "mean_root_target_projection_error_px": float(np.mean(root_errors)) if root_errors else None,
        "max_root_target_projection_error_px": float(np.max(root_errors)) if root_errors else None,
    }


def classify_failure(summary: dict[str, Any], *, area_min: float = 0.02, area_max: float = 0.18) -> list[str]:
    reasons: list[str] = []
    mean_area = summary.get("mean_robot_area_ratio")
    min_area = summary.get("min_robot_area_ratio")
    max_area = summary.get("max_robot_area_ratio")
    if summary.get("empty_robot_frames", 0):
        reasons.append("render_or_camera: robot mask is empty for at least one debug frame")
    if mean_area is not None and float(mean_area) < area_min:
        reasons.append("camera_or_base: robot is too small or mostly out of view")
    if max_area is not None and float(max_area) > area_max:
        reasons.append("scale_or_base: robot occupies too much of the image")
    if min_area is not None and float(min_area) < 1e-5:
        reasons.append("render_or_composite: zero visible robot area")
    if summary.get("max_center_area_ratio") is not None and float(summary["max_center_area_ratio"]) > 0.35:
        reasons.append("base_placement: robot/root intrudes into center of egocentric view")
    if summary.get("mean_visible_area_ratio") is not None and mean_area is not None:
        if float(mean_area) > 1e-5 and float(summary["mean_visible_area_ratio"]) < 0.35 * float(mean_area):
            reasons.append("depth_occlusion: depth compositing removes most robot pixels")
    if summary.get("mean_ee_projection_error_px") is not None and float(summary["mean_ee_projection_error_px"]) > 80.0:
        reasons.append("camera_convention_or_projection: EE projection is far from target")
    if summary.get("mean_target_to_robot_centroid_error_px") is not None and float(summary["mean_target_to_robot_centroid_error_px"]) > 160.0:
        reasons.append("camera_or_base: target midpoint is far from rendered robot centroid")
    if summary.get("mean_root_target_projection_error_px") is not None and float(summary["mean_root_target_projection_error_px"]) > 1000.0:
        reasons.append("base_placement: root/base projection is far outside the operation region")
    return reasons or ["pass_or_needs_visual_review"]


def chain_report(
    summary: dict[str, Any],
    *,
    area_min: float = 0.02,
    area_max: float = 0.18,
    max_ee_error_px: float = 80.0,
    max_centroid_error_px: float = 160.0,
    max_root_error_px: float = 1000.0,
) -> dict[str, Any]:
    mean_area = summary.get("mean_robot_area_ratio")
    max_area = summary.get("max_robot_area_ratio")
    mean_ee = summary.get("mean_ee_projection_error_px")
    mean_centroid = summary.get("mean_target_to_robot_centroid_error_px")
    mean_root = summary.get("mean_root_target_projection_error_px")
    converged = summary.get("render_converged_ratio")
    mean_pos = summary.get("render_mean_pos_error_m")
    mean_jaw = summary.get("render_mean_jaw_axis_error_deg")

    stages = {
        "retarget_targets_project_to_image": mean_ee is not None,
        "ik_tracks_targets": (
            converged is not None
            and float(converged) >= 0.95
            and mean_pos is not None
            and float(mean_pos) < 0.01
            and mean_jaw is not None
            and float(mean_jaw) < 25.0
        ),
        "ee_projects_to_target_pixels": mean_ee is not None and float(mean_ee) <= max_ee_error_px,
        "robot_visible": (
            summary.get("empty_robot_frames", 0) == 0
            and mean_area is not None
            and area_min <= float(mean_area)
            and max_area is not None
            and float(max_area) <= area_max
        ),
        "robot_body_near_operation_region": mean_centroid is not None and float(mean_centroid) <= max_centroid_error_px,
        "root_base_out_of_operation_region": mean_root is None or float(mean_root) <= max_root_error_px,
    }
    order = [
        "retarget_targets_project_to_image",
        "ik_tracks_targets",
        "ee_projects_to_target_pixels",
        "robot_visible",
        "robot_body_near_operation_region",
        "root_base_out_of_operation_region",
    ]
    failed = [name for name in order if not stages[name]]
    return {
        "stages": stages,
        "first_failed_stage": failed[0] if failed else None,
        "failed_stages": failed,
        "interpretation": (
            "full_chain_pass"
            if not failed
            else " -> ".join(order[: order.index(failed[0]) + 1])
        ),
    }


def projection_from_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    z = np.maximum(pts[:, 2], 1e-6)
    return np.column_stack([k[0, 0] * pts[:, 0] / z + k[0, 2], k[1, 1] * pts[:, 1] / z + k[1, 2]])


def finite_mean_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    lhs = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    n = min(len(lhs), len(rhs))
    if n == 0:
        return None
    lhs = lhs[:n]
    rhs = rhs[:n]
    valid = np.isfinite(lhs).all(axis=1) & np.isfinite(rhs).all(axis=1)
    if not np.any(valid):
        return None
    return float(np.linalg.norm(lhs[valid] - rhs[valid], axis=1).mean())


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
