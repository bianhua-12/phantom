from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT_ROOT = Path("/mnt/project_rlinf_hs/dreamzero_pretrain_data/22T_data/egodex_demo10_raw")
DEFAULT_OUTPUT_DIR = Path("outputs/qwenrobot_egodex")
_ROBOTTWIN_ALOHA_URDF = Path("/mnt/project_rlinf/jlchen/code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf")
DEFAULT_URDF = _ROBOTTWIN_ALOHA_URDF
DEFAULT_LEFT_EE_BODY = "fl_link6"
DEFAULT_RIGHT_EE_BODY = "fr_link6"
DEFAULT_LEFT_ROOT_BODY = "fl_link2"
DEFAULT_RIGHT_ROOT_BODY = "fr_link2"
DEFAULT_LEFT_EE_OFFSET = np.asarray([0.08457, 0.0, -0.00010349], dtype=np.float32)
DEFAULT_RIGHT_EE_OFFSET = np.asarray([0.08457, 0.0, -0.00010349], dtype=np.float32)
DEFAULT_FPS = 5.0

# EgoDex camera/world coordinates are converted into a z-up robot convention.
EGODEX_TO_ROBOT = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Demo:
    hdf5_path: Path
    video_path: Path
    rel_id: str


def natural_key(text: str) -> tuple[Any, ...]:
    tokens: list[Any] = []
    for tok in re.split(r"(\d+)", text):
        if tok:
            tokens.append(int(tok) if tok.isdigit() else tok)
    return tuple(tokens)


def safe_id(rel_id: str) -> str:
    return rel_id.replace("/", "__")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_demos(input_root: Path, max_episodes: int | None) -> list[Demo]:
    demos: list[Demo] = []
    for h5_path in sorted(input_root.rglob("*.hdf5"), key=lambda p: natural_key(str(p.relative_to(input_root)))):
        video_path = h5_path.with_suffix(".mp4")
        if not video_path.exists():
            continue
        rel_id = h5_path.relative_to(input_root).with_suffix("").as_posix()
        demos.append(Demo(hdf5_path=h5_path, video_path=video_path, rel_id=rel_id))
        if max_episodes is not None and len(demos) >= max_episodes:
            break
    if not demos:
        raise FileNotFoundError(f"No paired hdf5/mp4 demos under {input_root}")
    return demos


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": False, "frames": 0, "fps": 0.0, "width": 0, "height": 0}
    info = {
        "opened": True,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def frame_indices(num_frames: int, stride: int, max_frames: int | None) -> np.ndarray:
    indices = np.arange(0, int(num_frames), max(1, int(stride)), dtype=np.int64)
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    return indices


def write_sampled_video_and_frames(video_path: Path, indices: np.ndarray, video_out: Path, frames_dir: Path, fps: float) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {video_out}")
    for out_i, src_i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_i))
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        cv2.imwrite(str(frames_dir / f"{out_i:06d}.png"), frame)
    cap.release()
    writer.release()


def read_hdf5(path: Path) -> dict[str, Any]:
    keys = ["camera"]
    for side in ("left", "right"):
        keys.extend(
            [
                f"{side}Hand",
                f"{side}ThumbTip",
                f"{side}IndexFingerTip",
                f"{side}MiddleFingerTip",
                f"{side}Forearm",
                f"{side}Arm",
                f"{side}Shoulder",
            ]
        )
    with h5py.File(path, "r") as h5:
        transforms = {k: np.asarray(h5[f"transforms/{k}"], dtype=np.float32) for k in keys if f"transforms/{k}" in h5}
        confidences = {k: np.asarray(h5[f"confidences/{k}"], dtype=np.float32) for k in keys if f"confidences/{k}" in h5}
        intrinsic = np.asarray(h5["camera/intrinsic"], dtype=np.float32) if "camera/intrinsic" in h5 else None
        attrs = {k: jsonable(v) for k, v in h5.attrs.items()}
    return {"transforms": transforms, "confidences": confidences, "camera_intrinsic": intrinsic, "attrs": attrs}


def selected_instruction(attrs: dict[str, Any]) -> str:
    if str(attrs.get("which_llm_description", "1")) == "2":
        return str(attrs.get("llm_description2") or attrs.get("description2") or attrs.get("llm_description") or attrs.get("task") or "")
    return str(attrs.get("llm_description") or attrs.get("description") or attrs.get("task") or "")


def smooth_array(x: np.ndarray, window: int = 17) -> np.ndarray:
    if len(x) < 7:
        return x.astype(np.float32, copy=True)
    win = min(window, len(x) if len(x) % 2 else len(x) - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    return savgol_filter(x, window_length=win, polyorder=2, axis=0, mode="interp").astype(np.float32)


def smooth_rotations_gaussian_slerp(rotations: np.ndarray, sigma: float = 2.0, radius: int = 6) -> np.ndarray:
    if len(rotations) < 3:
        return rotations.astype(np.float32, copy=True)
    rotation_objs = R.from_matrix(rotations.astype(np.float64))
    out = []
    for i in range(len(rotations)):
        lo = max(0, i - radius)
        hi = min(len(rotations), i + radius + 1)
        offsets = np.arange(lo, hi, dtype=np.float64) - float(i)
        weights = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
        weights = weights / np.sum(weights)
        center = rotation_objs[i]
        rel = rotation_objs[lo:hi] * center.inv()
        avg_rotvec = np.sum(rel.as_rotvec() * weights[:, None], axis=0)
        out.append((R.from_rotvec(avg_rotvec) * center).as_matrix())
    return np.asarray(out, dtype=np.float32)


def retarget_hand(transforms: dict[str, np.ndarray], side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = transforms[f"{side}Hand"][:, :3, 3]
    thumb = transforms[f"{side}ThumbTip"][:, :3, 3]
    index = transforms[f"{side}IndexFingerTip"][:, :3, 3]
    middle = transforms[f"{side}MiddleFingerTip"][:, :3, 3]
    virtual_finger = 0.7 * index + 0.3 * middle
    width = np.linalg.norm(thumb - virtual_finger, axis=1)
    position = 0.5 * (thumb + virtual_finger)

    handedness_sign = 1.0 if side == "right" else -1.0
    z_axis = handedness_sign * (thumb - virtual_finger)
    z_axis = z_axis / np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1e-8)
    d_axis = virtual_finger - wrist
    y_axis = np.cross(z_axis, d_axis)
    y_axis = y_axis / np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-8)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=2)

    position = smooth_array(position)
    width = smooth_array(width[:, None])[:, 0]
    rotation = smooth_rotations_gaussian_slerp(rotation)
    return position.astype(np.float32), rotation.astype(np.float32), width.astype(np.float32)


def egodex_to_robot_pose(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aligned_position = (EGODEX_TO_ROBOT @ position.T).T
    aligned_rotation = EGODEX_TO_ROBOT[None, :, :] @ rotation
    return aligned_position.astype(np.float32), aligned_rotation.astype(np.float32)


def egodex_to_robot_points(position: np.ndarray) -> np.ndarray:
    return (EGODEX_TO_ROBOT @ position.T).T.astype(np.float32)


def egodex_camera_trajectory_to_robot(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = transforms[:, :3, 3]
    rotation = transforms[:, :3, :3]
    return egodex_to_robot_pose(position, rotation)


def egodex_world_to_camera_pose(position_world: np.ndarray, rotation_world: np.ndarray, camera_transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_position = camera_transforms[:, :3, 3]
    camera_rotation = camera_transforms[:, :3, :3]
    camera_rotation_t = np.swapaxes(camera_rotation, 1, 2)
    position_camera = np.einsum("nij,nj->ni", camera_rotation_t, position_world - camera_position)
    rotation_camera = np.einsum("nij,njk->nik", camera_rotation_t, rotation_world)
    return position_camera.astype(np.float32), rotation_camera.astype(np.float32)


def representative_bimanual_keyframes(left_xyz: np.ndarray, right_xyz: np.ndarray) -> np.ndarray:
    ids: set[int] = {len(left_xyz) // 2}
    for xyz in (left_xyz, right_xyz):
        for dim in range(3):
            ids.add(int(np.argmin(xyz[:, dim])))
            ids.add(int(np.argmax(xyz[:, dim])))
    return np.asarray(sorted(ids), dtype=np.int64)


def normalize_base_xyz_yaw(base: np.ndarray) -> np.ndarray:
    """Return canonical [x, y, z, yaw], accepting legacy [x, y, yaw]."""
    arr = np.asarray(base, dtype=np.float32).reshape(-1)
    if arr.shape[0] == 4:
        return arr.astype(np.float32)
    if arr.shape[0] == 3:
        return np.asarray([arr[0], arr[1], 0.0, arr[2]], dtype=np.float32)
    raise ValueError(f"Expected base pose [x,y,z,yaw] or legacy [x,y,yaw], got shape {arr.shape}")


def base_rotation_matrix(base_xyz_yaw: np.ndarray) -> np.ndarray:
    yaw = float(normalize_base_xyz_yaw(base_xyz_yaw)[3])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def transform_targets_to_base(points_world: np.ndarray, base_xyz_yaw: np.ndarray) -> np.ndarray:
    base = normalize_base_xyz_yaw(base_xyz_yaw)
    rot_t = base_rotation_matrix(base).T
    origin = base[:3].astype(np.float32)
    return (rot_t @ (points_world - origin).T).T.astype(np.float32)


def transform_rotations_to_base(rot_world: np.ndarray, base_xyz_yaw: np.ndarray) -> np.ndarray:
    rot_t = base_rotation_matrix(base_xyz_yaw).T
    return (rot_t[None, :, :] @ rot_world).astype(np.float32)


def transform_points_from_base(points_local: np.ndarray, base_xyz_yaw: np.ndarray) -> np.ndarray:
    base = normalize_base_xyz_yaw(base_xyz_yaw)
    rot = base_rotation_matrix(base)
    return ((rot @ points_local.T).T + base[:3].astype(np.float32)).astype(np.float32)


def save_video(frames: list[np.ndarray], output_video: Path, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_video), frames, fps=float(fps), quality=8, macro_block_size=1)
