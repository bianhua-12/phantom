# Qwen RobotManip Artifact Index

This repository contains many experiment outputs. This document marks the
minimum set worth keeping for reproducibility and diagnosis. It does not move or
delete any files.

## Minimum Keep Set

Keep these artifacts because they document the current baseline and failure
state:

- `phantom/qwenrobot/PHANTOM_EGODEX_BASELINE_REPORT.md`
  Historical fixed-root Phantom/Masquerade baseline report.
- `phantom/egodex_original/README.md`
  Commands and evidence for the original EgoDex-on-Phantom path.
- `phantom/egodex_original/IMPLEMENTATION_REPORT.md`
  More detailed implementation and smoke-test notes for the original path.
- `outputs/qwenrobot_egodex_episode0_shoulder/00_manifest.json`
  Sampled EgoDex jigsaw episode manifest for the Qwen-style path.
- `outputs/qwenrobot_egodex_episode0_shoulder/03_manifest.json`
  Existing explicit-base-search manifest. It shows feasibility but should not be
  interpreted as visual success.
- `outputs/qwenrobot_egodex_episode0_shoulder/04_manifest.json`
  MuJoCo raw robot render manifest.
- `outputs/qwenrobot_egodex_episode0_shoulder/06_aloha_camera_manifest.json`
  ALOHA camera-space render/IK experiment manifest.
- `outputs/qwenrobot_egodex_episode0_shoulder/06_aloha_camera_montage/extra__assemble_disassemble_jigsaw_puzzle__0.jpg`
  Representative montage for visual diagnosis.
- `outputs/qwenrobot_egodex_episode0_shoulder/07_aloha_replacement_manifest.json`
  Replacement/compositing experiment manifest.
- `outputs/qwenrobot_egodex_episode0_shoulder/07_aloha_replacement_montage/extra__assemble_disassemble_jigsaw_puzzle__0.jpg`
  Replacement montage documenting current quality.

## Useful But Not Canonical

These are useful for debugging but should not be used as final evidence:

- `outputs/qwenrobot_egodex_episode0_shoulder/04_raw_robot_overlay/`
  Raw overlay videos from the current explicit-base experiment.
- `outputs/qwenrobot_egodex_episode0_shoulder/04_robot_rgb/`
  Robot-only videos for mask and scale inspection.
- `outputs/qwenrobot_egodex_episode0_shoulder/05_phantom_fixed_panda_*`
  Fixed-camera Phantom/Panda probes. Keep only if comparing camera conventions
  or morphology.
- `outputs/qwenrobot_egodex_episode0_shoulder/_aloha_projection_diag_*.jpg`
  Projection diagnostics for selected frames.
- `outputs/qwenrobot_egodex_episode0_shoulder/_panda_projection_diag_*.jpg`
  Panda projection diagnostics for selected frames.
- `outputs/qwenrobot_egodex_episode0_shoulder/07_propainter_processed/`
  Current ProPainter processed background. It is evidence that ProPainter is
  wired, not evidence of final background quality.

## Historical Or Cleanup Candidates

The following classes of outputs can be moved to external storage or deleted
after confirming no active comparison depends on them:

- Repeated `outputs/kinova_camera_ik_*` runs that differ only by small axis,
  padding, depth, or width probes.
- Repeated `outputs/_base_offset_*` searches around Phantom shoulders.
- Skeleton overlay probes under `outputs/qwenrobot_skeleton_overlay_*`.
- Temporary single-frame render probes such as
  `outputs/qwenrobot_egodex_episode0_shoulder/phantom_original_single_frame_trials/`.

Do not delete these automatically in this pass. The cleanup decision should be a
separate storage-management step.

## Interpretation Rules

- A high valid ratio does not mean the robotized video is visually correct.
- Existing Phantom shoulders base-offset results are historical diagnostics, not
  the desired Qwen-style base search.
- The ProPainter jigsaw background is only a partial clean background; visible
  artifacts remain.
- The next trustworthy artifact should combine a corrected base-frame contract,
  orientation-aware IK feasibility, a non-black MP4, and a montage that verifies
  robot size, base visibility, gripper position, and jaw-axis alignment.
