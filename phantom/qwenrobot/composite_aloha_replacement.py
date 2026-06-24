from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import numpy as np

from phantom.qwenrobot.common import read_json, safe_id, video_info, write_json
from phantom.qwenrobot.prepare_egodex_for_phantom import (
    make_arm_masks,
    read_arm_points_2d,
    read_hand_sequence,
)
from phantom.qwenrobot.mechanical_cover_compositor import make_cover_layer


def read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")
    return np.asarray(frames, dtype=np.uint8)


def dilate_masks(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return masks.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return out


def feather_composite(raw: np.ndarray, fill: np.ndarray, masks: np.ndarray, feather: int) -> np.ndarray:
    blur_size = max(1, int(feather))
    if blur_size % 2 == 0:
        blur_size += 1
    out = np.empty_like(raw)
    for idx, (frame, fill_frame, mask) in enumerate(zip(raw, fill, masks)):
        alpha = mask.astype(np.float32)
        if blur_size > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
        alpha = np.clip(alpha[..., None], 0.0, 1.0)
        out[idx] = np.clip(frame.astype(np.float32) * (1.0 - alpha) + fill_frame.astype(np.float32) * alpha, 0, 255)
    return out.astype(np.uint8)


def temporal_nearest_fill(raw: np.ndarray, exclude_masks: np.ndarray) -> np.ndarray:
    n_frames, height, width = exclude_masks.shape
    fallback = np.median(raw, axis=0).astype(np.uint8)
    max_dist = np.iinfo(np.uint16).max

    prev_rgb = np.empty_like(raw)
    prev_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    last_rgb = fallback.copy()
    last_seen = np.full((height, width), -1, dtype=np.int32)
    for idx in range(n_frames):
        visible = ~exclude_masks[idx]
        last_rgb[visible] = raw[idx][visible]
        last_seen[visible] = idx
        prev_rgb[idx] = last_rgb
        dist = np.where(last_seen >= 0, idx - last_seen, max_dist)
        prev_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    next_rgb = np.empty_like(raw)
    next_dist = np.empty((n_frames, height, width), dtype=np.uint16)
    last_rgb = fallback.copy()
    next_seen = np.full((height, width), -1, dtype=np.int32)
    for idx in range(n_frames - 1, -1, -1):
        visible = ~exclude_masks[idx]
        last_rgb[visible] = raw[idx][visible]
        next_seen[visible] = idx
        next_rgb[idx] = last_rgb
        dist = np.where(next_seen >= 0, next_seen - idx, max_dist)
        next_dist[idx] = np.clip(dist, 0, max_dist).astype(np.uint16)

    return np.where((prev_dist <= next_dist)[..., None], prev_rgb, next_rgb)


def build_egodex_arm_data(
    traj: np.lib.npyio.NpzFile,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_indices = traj["frame_indices"].astype(np.int64)
    with h5py.File(str(traj["source_hdf5"]), "r") as h5:
        info = video_info(Path(str(traj["source_video"])))
        scale = np.asarray([width / info["width"], height / info["height"]], dtype=np.float32)
        _, left_kpts_2d, _ = read_hand_sequence(h5, "left", frame_indices)
        _, right_kpts_2d, _ = read_hand_sequence(h5, "right", frame_indices)
        arm_points_2d = read_arm_points_2d(h5, frame_indices)
    left_kpts_2d *= scale
    right_kpts_2d *= scale
    for side in arm_points_2d:
        arm_points_2d[side] *= scale
    masks = make_arm_masks(height, width, left_kpts_2d, right_kpts_2d, arm_points_2d).astype(bool)
    hand_points = {"left": left_kpts_2d, "right": right_kpts_2d}
    return masks, arm_points_2d, hand_points


def robot_mask_from_overlay(raw: np.ndarray, overlay: np.ndarray, threshold: float) -> np.ndarray:
    diff = np.linalg.norm(overlay.astype(np.int16) - raw.astype(np.int16), axis=-1)
    masks = diff > float(threshold)
    kernel = np.ones((5, 5), dtype=np.uint8)
    out = np.zeros_like(masks, dtype=bool)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    return out


def label(img: np.ndarray, text: str, idx: int) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (235, 48), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"f{idx}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def write_montage(path: Path, rows: list[tuple[str, np.ndarray]], frame_ids: list[int]) -> None:
    rendered_rows = []
    for name, frames in rows:
        cells = []
        for idx in frame_ids:
            if idx < 0 or idx >= len(frames):
                continue
            img = frames[idx]
            if img.ndim == 2:
                img = cv2.cvtColor((img.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cells.append(label(img, name, idx))
        if cells:
            rendered_rows.append(np.concatenate(cells, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.concatenate(rendered_rows, axis=0), quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite ALOHA robotized EgoDex video on a hand-free background.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--mask-dilation", type=int, default=21)
    parser.add_argument("--exclude-dilation", type=int, default=51)
    parser.add_argument("--feather", type=int, default=25)
    parser.add_argument("--robot-diff-threshold", type=float, default=12.0)
    parser.add_argument("--cover-link-radius", type=float, default=62.0)
    parser.add_argument("--cover-hand-radius", type=float, default=36.0)
    parser.add_argument("--clean-video", type=Path, default=None)
    args = parser.parse_args()

    manifest = read_json(args.output_dir / "00_manifest.json")
    aloha_manifest = read_json(args.output_dir / "06_aloha_camera_manifest.json")
    aloha_rows = {row["id"]: row for row in aloha_manifest["episodes"]}
    rows = []
    for row in manifest["episodes"][: args.max_episodes]:
        sid = safe_id(row["id"])
        if row["id"] not in aloha_rows:
            raise KeyError(f"No ALOHA render for {row['id']}")
        traj = np.load(row["trajectory_npz"], allow_pickle=True)
        raw = read_video_rgb(Path(row["sampled_video"]))
        overlay = read_video_rgb(Path(aloha_rows[row["id"]]["raw_robot_overlay_video"]))
        external_clean = read_video_rgb(args.clean_video.resolve()) if args.clean_video is not None else None
        n = min(len(raw), len(overlay), len(traj["frame_indices"]), len(external_clean) if external_clean is not None else len(raw))
        raw = raw[:n]
        overlay = overlay[:n]
        if external_clean is not None:
            external_clean = external_clean[:n]
        height, width = raw.shape[1:3]

        arm_masks, arm_points, hand_points = build_egodex_arm_data(traj, width, height)
        arm_masks = arm_masks[:n]
        for side in arm_points:
            arm_points[side] = arm_points[side][:n]
            hand_points[side] = hand_points[side][:n]
        composite_masks = dilate_masks(arm_masks, args.mask_dilation)
        if external_clean is None:
            exclude_masks = dilate_masks(arm_masks, args.exclude_dilation)
            fill = temporal_nearest_fill(raw, exclude_masks)
            clean = feather_composite(raw, fill, composite_masks, args.feather)
            clean_source = "temporal_nearest"
        else:
            if external_clean.shape[1:3] != raw.shape[1:3]:
                external_clean = np.asarray([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in external_clean])
            clean = external_clean
            clean_source = str(args.clean_video)
        robot_masks = robot_mask_from_overlay(raw, overlay, args.robot_diff_threshold)

        final = clean.copy()
        scale = height / 1080.0
        link_radius = max(10, int(round(args.cover_link_radius * scale)))
        hand_radius = max(8, int(round(args.cover_hand_radius * scale)))
        cover_masks = np.zeros(robot_masks.shape, dtype=bool)
        for idx in range(n):
            cover = make_cover_layer(final[idx].shape, arm_points, hand_points, idx, link_radius, hand_radius)
            cover_mask = (cover.sum(axis=-1) > 0) & composite_masks[idx]
            cover_masks[idx] = cover_mask
            final[idx][cover_mask] = cover[cover_mask]
        final[robot_masks] = overlay[robot_masks]

        video_path = args.output_dir / "07_aloha_replacement" / f"{sid}.mp4"
        montage_path = args.output_dir / "07_aloha_replacement_montage" / f"{sid}.jpg"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(row.get("output_fps", 5.0))
        iio.imwrite(
            video_path,
            final,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )
        frame_ids = np.linspace(0, n - 1, min(5, n), dtype=int).tolist()
        write_montage(
            montage_path,
            [
                ("raw", raw),
                ("human_mask", composite_masks),
                ("clean_bg", clean),
                ("mechanical_cover", np.where(cover_masks[..., None], final, clean)),
                ("robot_raw_overlay", overlay),
                ("replacement", final),
            ],
            frame_ids,
        )
        rows.append(
            {
                "id": row["id"],
                "replacement_video": video_path,
                "montage": montage_path,
                "frames": int(n),
                "fps": fps,
                "mask_mean_area": float(arm_masks.mean()),
                "composite_mask_mean_area": float(composite_masks.mean()),
                "robot_mask_mean_area": float(robot_masks.mean()),
                "cover_mask_mean_area": float(cover_masks.mean()),
                "clean_source": clean_source,
            }
        )
    write_json(args.output_dir / "07_aloha_replacement_manifest.json", {"stage": "aloha_replacement_temporal_nearest", "episodes": rows})
    print(args.output_dir / "07_aloha_replacement_manifest.json")


if __name__ == "__main__":
    main()
