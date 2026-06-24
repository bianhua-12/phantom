from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from phantom.qwenrobot.common import (
    DEFAULT_LEFT_EE_OFFSET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RIGHT_EE_OFFSET,
    DEFAULT_URDF,
    read_json,
    safe_id,
    save_video,
    video_info,
    write_json,
)
from phantom.qwenrobot.mujoco_utils import (
    active_dof_mask,
    body_id,
    body_point,
    clamp_qpos,
    ik_position,
    load_model,
    set_parallel_gripper_width,
    style_robot_geoms,
)
from phantom.qwenrobot.render_phantom_fixed_panda import (
    fixed_root_from_forearm_border,
    load_body_points_camera,
    read_video_frames,
)


def fovy_from_intrinsic(intrinsic: np.ndarray, height: int) -> float:
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * float(intrinsic[1, 1]))))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def ik_position_with_link_targets(
    mujoco_module,
    model,
    data,
    *,
    ee_target: np.ndarray,
    ee_body_id: int,
    ee_offset: np.ndarray,
    link_targets: list[tuple[int, np.ndarray, float]],
    initial_qpos: np.ndarray | None,
    tol: float,
    max_iters: int,
    active_joint_prefixes: tuple[str, ...],
) -> tuple[bool, np.ndarray, float, float]:
    """Solve IK while softly pulling intermediate links toward human arm anchors."""
    data.qpos[:] = 0.0 if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64)
    clamp_qpos(model, data.qpos)
    mujoco_module.mj_forward(model, data)
    ee_target = np.asarray(ee_target, dtype=np.float64).reshape(3)
    ee_offset = np.asarray(ee_offset, dtype=np.float64).reshape(3)
    dof_mask = active_dof_mask(mujoco_module, model, active_joint_prefixes)
    damping = 1e-4
    for _ in range(max_iters):
        rows = []
        residuals = []
        ee_pos = body_point(data, ee_body_id, ee_offset)
        ee_err = ee_target - ee_pos
        ee_err_norm = float(np.linalg.norm(ee_err))
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco_module.mj_jac(model, data, jacp, jacr, ee_pos, ee_body_id)
        rows.append(jacp)
        residuals.append(ee_err)

        aux_sq = 0.0
        aux_weight_sum = 0.0
        for link_body_id, target, weight in link_targets:
            if weight <= 0.0:
                continue
            target = np.asarray(target, dtype=np.float64).reshape(3)
            link_pos = body_point(data, link_body_id, np.zeros(3, dtype=np.float64))
            err = target - link_pos
            aux_sq += float(weight) * float(np.dot(err, err))
            aux_weight_sum += float(weight)
            jacp_link = np.zeros((3, model.nv), dtype=np.float64)
            jacr_link = np.zeros((3, model.nv), dtype=np.float64)
            mujoco_module.mj_jac(model, data, jacp_link, jacr_link, link_pos, link_body_id)
            rows.append(float(weight) * jacp_link)
            residuals.append(float(weight) * err)

        aux_err = math.sqrt(aux_sq / max(aux_weight_sum, 1e-8)) if aux_weight_sum > 0.0 else 0.0
        if ee_err_norm < tol:
            return True, data.qpos.copy(), ee_err_norm, aux_err

        jac = np.vstack(rows)
        residual = np.concatenate(residuals)
        lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(lhs, residual)
        dq = dq * dof_mask
        mujoco_module.mj_integratePos(model, data.qpos, dq, 0.35)
        clamp_qpos(model, data.qpos)
        mujoco_module.mj_forward(model, data)

    ee_err_norm = float(np.linalg.norm(ee_target - body_point(data, ee_body_id, ee_offset)))
    aux_errors = []
    for link_body_id, target, weight in link_targets:
        if weight > 0.0:
            aux_errors.append(float(np.linalg.norm(np.asarray(target, dtype=np.float64) - body_point(data, link_body_id, np.zeros(3)))))
    aux_err = float(np.mean(aux_errors)) if aux_errors else 0.0
    return ee_err_norm < tol, data.qpos.copy(), ee_err_norm, aux_err


def project_camera_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = np.maximum(points[:, 2], 1e-6)
    return np.column_stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy])


def base_quat_x_toward_target(root: np.ndarray, target: np.ndarray, side: str) -> np.ndarray:
    x_axis = normalize(target - root)
    y_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if side == "right":
        y_hint = -y_hint
    z_axis = np.cross(x_axis, y_hint)
    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.cross(x_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    z_axis = normalize(z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat(scalar_first=True)


def render_rgb_and_mask(renderer, model, data, camera) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    scene_camera = renderer._scene.camera[0]
    scene_camera.pos[:] = [0.0, 0.0, 0.0]
    scene_camera.forward[:] = [0.0, 0.0, 1.0]
    scene_camera.up[:] = [0.0, -1.0, 0.0]
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    scene_camera = renderer._scene.camera[0]
    scene_camera.pos[:] = [0.0, 0.0, 0.0]
    scene_camera.forward[:] = [0.0, 0.0, 1.0]
    scene_camera.up[:] = [0.0, -1.0, 0.0]
    seg = renderer.render()
    geom_ids = seg[:, :, 0].astype(np.int32)
    mask = np.zeros(geom_ids.shape, dtype=bool)
    for geom_id in np.unique(geom_ids):
        if geom_id < 0 or geom_id >= model.ngeom:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("fl_") or body_name.startswith("fr_"):
            mask |= geom_ids == geom_id
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return rgb, mask


def overlay_robot(raw: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[robot_mask] = robot_rgb[robot_mask]
    return out


def draw_link_segment(layer: np.ndarray, a: np.ndarray, b: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return
    p0 = tuple(np.round(a).astype(int).tolist())
    p1 = tuple(np.round(b).astype(int).tolist())
    cv2.line(layer, p0, p1, (22, 24, 26), radius + 12, cv2.LINE_AA)
    cv2.line(layer, p0, p1, color, radius, cv2.LINE_AA)
    cv2.line(layer, p0, p1, (238, 240, 240), max(3, radius // 5), cv2.LINE_AA)


def draw_joint(layer: np.ndarray, point: np.ndarray, radius: int) -> None:
    if not np.isfinite(point).all():
        return
    center = tuple(np.round(point).astype(int).tolist())
    cv2.circle(layer, center, radius + 8, (18, 20, 22), -1, cv2.LINE_AA)
    cv2.circle(layer, center, radius, (70, 74, 78), -1, cv2.LINE_AA)
    cv2.circle(layer, center, max(4, radius // 3), (220, 224, 224), -1, cv2.LINE_AA)


def human_link_targets(
    body_points_camera: dict[str, dict[str, np.ndarray]],
    side: str,
    frame_i: int,
    *,
    root: np.ndarray,
    link2_body_id: int,
    link4_body_id: int,
    link5_body_id: int,
    weights: tuple[float, float, float],
) -> list[tuple[int, np.ndarray, float]]:
    forearm = body_points_camera[side]["Forearm"][frame_i]
    hand = body_points_camera[side]["Hand"][frame_i]
    if float(forearm[2]) < 0.08:
        forearm = 0.45 * forearm + 0.55 * hand
        forearm[2] = max(float(forearm[2]), 0.08)
    link2_target = 0.55 * np.asarray(root, dtype=np.float64) + 0.45 * forearm
    return [
        (link2_body_id, link2_target, weights[0]),
        (link4_body_id, forearm, weights[1]),
        (link5_body_id, hand, weights[2]),
    ]


def draw_kinematic_proxy(
    frame: np.ndarray,
    mujoco_module,
    model,
    data,
    intrinsic: np.ndarray,
    *,
    radius_px: int,
) -> np.ndarray:
    layer = np.zeros_like(frame)
    for prefix, color in (("fl", (196, 202, 202)), ("fr", (212, 216, 216))):
        pts_3d = []
        for idx in range(1, 7):
            name = f"{prefix}_link{idx}"
            bid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            pts_3d.append(data.xpos[bid].copy())
        if len(pts_3d) < 2:
            continue
        pts_3d_arr = np.asarray(pts_3d, dtype=np.float64)
        pts_2d = project_camera_points(pts_3d_arr, intrinsic)
        valid = pts_3d_arr[:, 2] > 0.05
        for a, b, ok_a, ok_b in zip(pts_2d[:-1], pts_2d[1:], valid[:-1], valid[1:]):
            if ok_a and ok_b:
                draw_link_segment(layer, a, b, radius_px, color)
        for p, ok in zip(pts_2d[1:-1], valid[1:-1]):
            if ok:
                draw_joint(layer, p, max(9, radius_px // 2))
        if valid[-1]:
            draw_joint(layer, pts_2d[-1], max(10, radius_px // 2))
    mask = layer.sum(axis=-1) > 0
    out = frame.copy()
    out[mask] = layer[mask]
    return out


def make_montage(frames: list[np.ndarray], output_path: Path) -> None:
    if not frames:
        return
    ids = [0, len(frames) // 2, len(frames) - 1]
    montage = np.concatenate([frames[i] for i in ids], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Aloha/ARX front arms directly in EgoDex camera space.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--ik-tol", type=float, default=0.025)
    parser.add_argument("--root-border-pad-px", type=float, default=120.0)
    parser.add_argument("--root-depth-offset-m", type=float, default=0.12)
    parser.add_argument("--left-ee-body", type=str, default="fl_link6")
    parser.add_argument("--right-ee-body", type=str, default="fr_link6")
    parser.add_argument("--left-root-body", type=str, default="fl_link1")
    parser.add_argument("--right-root-body", type=str, default="fr_link1")
    parser.add_argument("--left-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_LEFT_EE_OFFSET))
    parser.add_argument("--right-ee-offset", type=float, nargs=3, default=tuple(float(v) for v in DEFAULT_RIGHT_EE_OFFSET))
    parser.add_argument("--proxy-link-radius-px", type=int, default=42)
    parser.add_argument("--link-target-weight", type=float, default=0.18)
    parser.add_argument("--forearm-target-weight", type=float, default=0.35)
    parser.add_argument("--hand-target-weight", type=float, default=0.16)
    parser.add_argument("--disable-link-targets", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.output_dir / "00_manifest.json")
    rows = []
    for row in manifest["episodes"][: args.max_episodes]:
        sid = safe_id(row["id"])
        traj = np.load(row["trajectory_npz"], allow_pickle=True)
        raw_frames = read_video_frames(Path(row["sampled_video"]))
        info = video_info(Path(row["sampled_video"]))
        width, height = int(info["width"]), int(info["height"])
        intrinsic = traj["camera_intrinsic"].astype(np.float64)
        body_points_camera = {
            side: {
                part: load_body_points_camera(traj, side, part)
                for part in ("Arm", "Forearm", "Hand")
            }
            for side in ("left", "right")
        }
        missing = [
            f"{side}.{part}"
            for side, values in body_points_camera.items()
            for part, points in values.items()
            if points is None
        ]
        if missing:
            raise ValueError(f"Missing camera-space body points for {sid}: {missing}")

        mujoco_module, model, data = load_model(args.urdf)
        style_robot_geoms(mujoco_module, model)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        model.vis.global_.fovy = fovy_from_intrinsic(intrinsic, height)

        left_root, left_root_pixel = fixed_root_from_forearm_border(
            traj["left_ee_pos_camera"].astype(np.float64),
            load_body_points_camera(traj, "left", "Forearm"),
            load_body_points_camera(traj, "left", "Arm"),
            intrinsic,
            width,
            height,
            "left",
            border_pad_px=args.root_border_pad_px,
            depth_offset_m=args.root_depth_offset_m,
        )
        right_root, right_root_pixel = fixed_root_from_forearm_border(
            traj["right_ee_pos_camera"].astype(np.float64),
            load_body_points_camera(traj, "right", "Forearm"),
            load_body_points_camera(traj, "right", "Arm"),
            intrinsic,
            width,
            height,
            "right",
            border_pad_px=args.root_border_pad_px,
            depth_offset_m=args.root_depth_offset_m,
        )
        left_target_median = np.median(traj["left_ee_pos_camera"].astype(np.float64), axis=0)
        right_target_median = np.median(traj["right_ee_pos_camera"].astype(np.float64), axis=0)
        left_root_id = body_id(mujoco_module, model, args.left_root_body)
        right_root_id = body_id(mujoco_module, model, args.right_root_body)
        model.body_pos[left_root_id] = left_root
        model.body_quat[left_root_id] = base_quat_x_toward_target(left_root, left_target_median, "left")
        model.body_pos[right_root_id] = right_root
        model.body_quat[right_root_id] = base_quat_x_toward_target(right_root, right_target_median, "right")

        left_ee_id = body_id(mujoco_module, model, args.left_ee_body)
        right_ee_id = body_id(mujoco_module, model, args.right_ee_body)
        left_link2_id = body_id(mujoco_module, model, "fl_link2")
        left_link4_id = body_id(mujoco_module, model, "fl_link4")
        left_link5_id = body_id(mujoco_module, model, "fl_link5")
        right_link2_id = body_id(mujoco_module, model, "fr_link2")
        right_link4_id = body_id(mujoco_module, model, "fr_link4")
        right_link5_id = body_id(mujoco_module, model, "fr_link5")
        link_weights = (
            0.0 if args.disable_link_targets else float(args.link_target_weight),
            0.0 if args.disable_link_targets else float(args.forearm_target_weight),
            0.0 if args.disable_link_targets else float(args.hand_target_weight),
        )
        renderer = mujoco_module.Renderer(model, height=height, width=width)
        camera = mujoco_module.MjvCamera()
        camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
        data.qpos[:] = 0.0
        mujoco_module.mj_forward(model, data)
        previous_qpos = data.qpos.copy()

        robot_frames: list[np.ndarray] = []
        overlay_frames: list[np.ndarray] = []
        qpos_rows = []
        ik_errors = []
        link_aux_errors = []
        num_frames = min(len(raw_frames), len(traj["left_ee_pos_camera"]), len(traj["right_ee_pos_camera"]))
        for frame_i in tqdm(range(num_frames), desc=f"Render Aloha {sid}"):
            left_target = traj["left_ee_pos_camera"][frame_i].astype(np.float64)
            right_target = traj["right_ee_pos_camera"][frame_i].astype(np.float64)
            if args.disable_link_targets:
                ok_left, qpos, err_left = ik_position(
                    mujoco_module,
                    model,
                    data,
                    left_target,
                    left_ee_id,
                    ee_offset=np.asarray(args.left_ee_offset, dtype=np.float64),
                    initial_qpos=previous_qpos,
                    tol=args.ik_tol,
                    max_iters=180,
                    active_joint_prefixes=("fl_joint",),
                    orientation_weight=0.0,
                )
                aux_left = 0.0
                ok_right, qpos, err_right = ik_position(
                    mujoco_module,
                    model,
                    data,
                    right_target,
                    right_ee_id,
                    ee_offset=np.asarray(args.right_ee_offset, dtype=np.float64),
                    initial_qpos=qpos,
                    tol=args.ik_tol,
                    max_iters=180,
                    active_joint_prefixes=("fr_joint",),
                    orientation_weight=0.0,
                )
                aux_right = 0.0
            else:
                ok_left, qpos, err_left, aux_left = ik_position_with_link_targets(
                    mujoco_module,
                    model,
                    data,
                    ee_target=left_target,
                    ee_body_id=left_ee_id,
                    ee_offset=np.asarray(args.left_ee_offset, dtype=np.float64),
                    link_targets=human_link_targets(
                        body_points_camera,
                        "left",
                        frame_i,
                        root=left_root,
                        link2_body_id=left_link2_id,
                        link4_body_id=left_link4_id,
                        link5_body_id=left_link5_id,
                        weights=link_weights,
                    ),
                    initial_qpos=previous_qpos,
                    tol=args.ik_tol,
                    max_iters=220,
                    active_joint_prefixes=("fl_joint",),
                )
                ok_right, qpos, err_right, aux_right = ik_position_with_link_targets(
                    mujoco_module,
                    model,
                    data,
                    ee_target=right_target,
                    ee_body_id=right_ee_id,
                    ee_offset=np.asarray(args.right_ee_offset, dtype=np.float64),
                    link_targets=human_link_targets(
                        body_points_camera,
                        "right",
                        frame_i,
                        root=right_root,
                        link2_body_id=right_link2_id,
                        link4_body_id=right_link4_id,
                        link5_body_id=right_link5_id,
                        weights=link_weights,
                    ),
                    initial_qpos=qpos,
                    tol=args.ik_tol,
                    max_iters=220,
                    active_joint_prefixes=("fr_joint",),
                )
            data.qpos[:] = qpos
            set_parallel_gripper_width(mujoco_module, model, data, "fl", float(traj["left_gripper_width"][frame_i]))
            set_parallel_gripper_width(mujoco_module, model, data, "fr", float(traj["right_gripper_width"][frame_i]))
            mujoco_module.mj_forward(model, data)
            previous_qpos = data.qpos.copy()
            qpos_rows.append(previous_qpos)
            ik_errors.append([float(err_left), float(err_right), bool(ok_left), bool(ok_right)])
            link_aux_errors.append([float(aux_left), float(aux_right)])
            rgb, mask = render_rgb_and_mask(renderer, model, data, camera)
            robot_frames.append(rgb)
            overlay = draw_kinematic_proxy(
                raw_frames[frame_i],
                mujoco_module,
                model,
                data,
                intrinsic,
                radius_px=args.proxy_link_radius_px,
            )
            overlay_frames.append(overlay_robot(overlay, rgb, mask))
        renderer.close()

        robot_video = args.output_dir / "06_aloha_camera_robot" / f"{sid}.mp4"
        overlay_video = args.output_dir / "06_aloha_camera_overlay" / f"{sid}.mp4"
        qpos_npz = args.output_dir / "06_aloha_camera_qpos" / f"{sid}.npz"
        montage_path = args.output_dir / "06_aloha_camera_montage" / f"{sid}.jpg"
        save_video(robot_frames, robot_video, 5.0)
        save_video(overlay_frames, overlay_video, 5.0)
        make_montage(overlay_frames, montage_path)
        qpos_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            qpos_npz,
            qpos=np.asarray(qpos_rows, dtype=np.float32),
            ik_errors=np.asarray(ik_errors, dtype=np.float32),
            link_aux_errors=np.asarray(link_aux_errors, dtype=np.float32),
            left_root=left_root.astype(np.float32),
            right_root=right_root.astype(np.float32),
            left_root_pixel=np.asarray(left_root_pixel, dtype=np.float32),
            right_root_pixel=np.asarray(right_root_pixel, dtype=np.float32),
            urdf=str(args.urdf),
        )
        rows.append(
            {
                "id": row["id"],
                "robot_video": robot_video,
                "raw_robot_overlay_video": overlay_video,
                "qpos_npz": qpos_npz,
                "montage": montage_path,
                "urdf": args.urdf,
                "left_root_camera": left_root,
                "right_root_camera": right_root,
                "left_root_pixel": left_root_pixel,
                "right_root_pixel": right_root_pixel,
                "mean_left_ik_error": float(np.mean([v[0] for v in ik_errors])),
                "mean_right_ik_error": float(np.mean([v[1] for v in ik_errors])),
                "failed_left_frames": int(np.sum([not v[2] for v in ik_errors])),
                "failed_right_frames": int(np.sum([not v[3] for v in ik_errors])),
                "mean_left_link_aux_error": float(np.mean([v[0] for v in link_aux_errors])),
                "mean_right_link_aux_error": float(np.mean([v[1] for v in link_aux_errors])),
                "link_target_weights": link_weights,
                "proxy_link_radius_px": args.proxy_link_radius_px,
            }
        )
    write_json(args.output_dir / "06_aloha_camera_manifest.json", {"stage": "aloha_camera_space_forearm_roots_link_target_ik", "episodes": rows})
    print(args.output_dir / "06_aloha_camera_manifest.json")


if __name__ == "__main__":
    main()
