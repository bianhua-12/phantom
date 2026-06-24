from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial.transform import Rotation

QWEN_TO_PHANTOM_TOOL_LEGACY = (
    Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    @ Rotation.from_euler("z", -135.0, degrees=True).as_matrix()
)


@dataclass(frozen=True)
class FrameCandidate:
    name: str
    matrix: np.ndarray
    qwen_jaw_axis_local: np.ndarray
    phantom_jaw_axis_local: np.ndarray


def signed_permutation_candidates() -> list[FrameCandidate]:
    """All proper-rotation signed permutations for Qwen-local to Phantom-local axes."""
    candidates: list[FrameCandidate] = []
    basis = np.eye(3, dtype=np.float64)
    qwen_jaw = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    axis_names = ("x", "y", "z")
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.column_stack([signs[i] * basis[perm[i]] for i in range(3)])
            if np.linalg.det(matrix) < 0.5:
                continue
            parts = []
            for dst_axis, src_axis in enumerate(perm):
                sign = "+" if signs[dst_axis] > 0 else "-"
                parts.append(f"p{axis_names[dst_axis]}={sign}q{axis_names[src_axis]}")
            phantom_jaw = matrix.T @ qwen_jaw
            candidates.append(
                FrameCandidate(
                    ",".join(parts),
                    matrix.astype(np.float64),
                    qwen_jaw_axis_local=qwen_jaw.copy(),
                    phantom_jaw_axis_local=phantom_jaw.astype(np.float64),
                )
            )
    return candidates


def recover_qwen_rotations_from_legacy(rotations: np.ndarray) -> np.ndarray:
    return np.asarray(rotations, dtype=np.float64) @ QWEN_TO_PHANTOM_TOOL_LEGACY.T


def apply_frame_candidate(rotations: np.ndarray, candidate: FrameCandidate) -> np.ndarray:
    return np.asarray(rotations, dtype=np.float64) @ candidate.matrix
