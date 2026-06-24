from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phantom.qwenrobot.canonical_render import (
    InputValidationError,
    resolve_source_hdf5,
    validate_hdf5_source,
    write_failed_validation,
)
from phantom.qwenrobot.composite_canonical import (
    CompositeInputError,
    assert_not_invalid_artifact,
    compose_depth_overlay,
)


def test_hdf5_validation_failure_writes_only_failure_marker(tmp_path: Path) -> None:
    bad_hdf5 = tmp_path / "not_hdf5.hdf5"
    bad_hdf5.write_text("not an hdf5 file", encoding="utf-8")
    output_dir = tmp_path / "out"

    with pytest.raises(InputValidationError):
        validate_hdf5_source(bad_hdf5)

    failure = write_failed_validation(output_dir, InputValidationError("bad hdf5"), {"source_hdf5_arg": str(bad_hdf5)})
    assert failure.name == "FAILED_INPUT_VALIDATION.json"
    assert failure.exists()
    assert not (output_dir / "robot_render_canonical.npz").exists()


def test_source_hdf5_resolution_prefers_cli_then_manifest_then_trajectory(tmp_path: Path) -> None:
    trajectory = tmp_path / "traj.npz"
    np.savez_compressed(trajectory, source_hdf5=str(tmp_path / "from_traj.hdf5"))
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "adapter_manifest.json").write_text(
        '{"source_hdf5": "%s"}' % str(tmp_path / "from_manifest.hdf5"),
        encoding="utf-8",
    )

    explicit, origin = resolve_source_hdf5(
        explicit_source_hdf5=tmp_path / "from_cli.hdf5",
        processed_demo_dir=processed,
        trajectory_npz=trajectory,
    )
    assert explicit.name == "from_cli.hdf5"
    assert origin == "cli"

    manifest, origin = resolve_source_hdf5(
        explicit_source_hdf5=None,
        processed_demo_dir=processed,
        trajectory_npz=trajectory,
    )
    assert manifest.name == "from_manifest.hdf5"
    assert origin == "adapter_manifest"

    (processed / "adapter_manifest.json").unlink()
    traj, origin = resolve_source_hdf5(
        explicit_source_hdf5=None,
        processed_demo_dir=processed,
        trajectory_npz=trajectory,
    )
    assert traj.name == "from_traj.hdf5"
    assert origin == "trajectory_npz"


def test_depth_composite_uses_canonical_mask_and_depth_formula() -> None:
    background = np.zeros((2, 3, 3), dtype=np.uint8)
    robot_rgb = np.full((2, 3, 3), 200, dtype=np.uint8)
    robot_mask = np.asarray(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    robot_depth = np.asarray(
        [
            [0.5, 1.5, 0.5],
            [2.0, 0.0, np.inf],
        ],
        dtype=np.float32,
    )
    scene_depth = np.asarray(
        [
            [0.6, 1.0, 0.1],
            [2.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )

    overlay, visible, occluded = compose_depth_overlay(
        background,
        robot_rgb,
        robot_mask,
        robot_depth,
        scene_depth,
        margin=0.0,
    )

    expected_visible = np.asarray(
        [
            [True, False, False],
            [True, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(visible, expected_visible)
    np.testing.assert_array_equal(occluded, robot_mask & ~expected_visible)
    assert np.all(overlay[expected_visible] == 200)
    assert np.all(overlay[robot_mask & ~expected_visible] == 0)


def test_composite_rejects_invalid_no_hdf5_artifacts(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "explicit_ik_render_invalid_no_hdf5"
    invalid_dir.mkdir()
    path = invalid_dir / "robot_render_canonical.npz"
    path.write_bytes(b"placeholder")

    with pytest.raises(CompositeInputError):
        assert_not_invalid_artifact(path)

    marker_dir = tmp_path / "render"
    marker_dir.mkdir()
    (marker_dir / "INVALID_NO_HDF5.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompositeInputError):
        assert_not_invalid_artifact(marker_dir / "robot_render_canonical.npz")
