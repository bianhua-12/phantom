from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phantom.qwenrobot.common import (
    DEFAULT_FPS,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_DIR,
    discover_demos,
    egodex_camera_trajectory_to_robot,
    egodex_to_robot_points,
    egodex_to_robot_pose,
    egodex_world_to_camera_pose,
    frame_indices,
    read_hdf5,
    safe_id,
    selected_instruction,
    video_info,
    write_json,
    write_sampled_video_and_frames,
    retarget_hand,
)


def _aligned_points(transforms: dict[str, np.ndarray], key: str, indices: np.ndarray) -> np.ndarray | None:
    if key not in transforms:
        return None
    return egodex_to_robot_points(transforms[key][:, :3, 3])[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget EgoDex hand keypoints with QwenRobot equations (1)-(2).")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args()

    rows = []
    for demo in discover_demos(args.input_root, args.max_episodes):
        sid = safe_id(demo.rel_id)
        h5 = read_hdf5(demo.hdf5_path)
        info = video_info(demo.video_path)
        right_pos_world, right_rot_world, right_width = retarget_hand(h5["transforms"], "right")
        left_pos_world, left_rot_world, left_width = retarget_hand(h5["transforms"], "left")
        right_pos, right_rot = egodex_to_robot_pose(right_pos_world, right_rot_world)
        left_pos, left_rot = egodex_to_robot_pose(left_pos_world, left_rot_world)
        if "camera" in h5["transforms"]:
            right_pos_camera, right_rot_camera = egodex_world_to_camera_pose(right_pos_world, right_rot_world, h5["transforms"]["camera"])
            left_pos_camera, left_rot_camera = egodex_world_to_camera_pose(left_pos_world, left_rot_world, h5["transforms"]["camera"])
            cam_pos, cam_rot = egodex_camera_trajectory_to_robot(h5["transforms"]["camera"])
        else:
            right_pos_camera, right_rot_camera = right_pos_world, right_rot_world
            left_pos_camera, left_rot_camera = left_pos_world, left_rot_world
            cam_pos, cam_rot = None, None

        n = min(len(right_pos), len(left_pos), int(info["frames"]))
        indices = frame_indices(n, args.frame_stride, args.max_frames)
        video_out = args.output_dir / "00_sampled_videos" / f"{sid}.mp4"
        frames_dir = args.output_dir / "00_sampled_frames" / sid
        write_sampled_video_and_frames(demo.video_path, indices, video_out, frames_dir, args.output_fps)

        traj_path = args.output_dir / "00_ee_trajectories" / f"{sid}.npz"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_indices": indices,
            "left_ee_pos": left_pos[indices],
            "left_ee_rot": left_rot[indices],
            "left_ee_pos_camera": left_pos_camera[indices],
            "left_ee_rot_camera": left_rot_camera[indices],
            "left_ee_pos_world": left_pos_world[indices],
            "left_ee_rot_world": left_rot_world[indices],
            "left_gripper_width": left_width[indices],
            "right_ee_pos": right_pos[indices],
            "right_ee_rot": right_rot[indices],
            "right_ee_pos_camera": right_pos_camera[indices],
            "right_ee_rot_camera": right_rot_camera[indices],
            "right_ee_pos_world": right_pos_world[indices],
            "right_ee_rot_world": right_rot_world[indices],
            "right_gripper_width": right_width[indices],
            "camera_pos": cam_pos[indices] if cam_pos is not None else None,
            "camera_rot": cam_rot[indices] if cam_rot is not None else None,
            "camera_intrinsic": h5["camera_intrinsic"],
            "coordinate_frame": "egodex_world_aligned_to_robot_z_up",
            "source_hdf5": str(demo.hdf5_path),
            "source_video": str(demo.video_path),
            "rel_id": demo.rel_id,
            "action_alignment": "qwenrobot_equations_1_2",
        }
        for side in ("left", "right"):
            for part in ("Hand", "Forearm", "Arm", "Shoulder"):
                value = _aligned_points(h5["transforms"], f"{side}{part}", indices)
                if value is not None:
                    payload[f"{side}_{part.lower()}_pos"] = value
        np.savez_compressed(traj_path, **payload)

        rows.append(
            {
                "id": demo.rel_id,
                "trajectory_npz": traj_path,
                "sampled_video": video_out,
                "sampled_frames_dir": frames_dir,
                "instruction": selected_instruction(h5["attrs"]),
                "source_frames": int(info["frames"]),
                "sampled_frames": int(len(indices)),
                "frame_stride": int(args.frame_stride),
                "output_fps": float(args.output_fps),
                "has_camera_trajectory": bool(cam_pos is not None),
                "hands": ["left", "right"],
            }
        )
    write_json(args.output_dir / "00_manifest.json", {"stage": "qwenrobot_egodex_retarget", "episodes": rows})
    print(args.output_dir / "00_manifest.json")

if __name__ == "__main__":
    main()
