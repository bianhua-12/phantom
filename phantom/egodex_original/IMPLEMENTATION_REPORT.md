# EgoDex Phantom Original Reproduction Report

## Scope

This path is the current baseline for adapting EgoDex to the original
Phantom/Masquerade repository. It intentionally keeps the robotization stages
from the original repo:

- `ActionProcessor`
- `SmoothingProcessor`
- `HandInpaintProcessor` with E2FGVI
- `RobotInpaintProcessor`
- `Kinova3` + `Robotiq85`
- bimanual `shoulders` setup
- original Masquerade/EPIC fixed camera extrinsics

The only dataset-specific layer is the EgoDex adapter that writes Phantom's
expected files from EgoDex `.hdf5` and `.mp4`.

## What The Adapter Replaces

The original Masquerade EPIC path expects EPIC-specific intermediate files.
EgoDex does not provide those exact files, but it does provide stronger hand
and arm annotations in HDF5.

The adapter writes these Phantom-compatible files directly:

- `video_L.mp4`, `video_R.mp4`, `video_rgb_imgs.mkv`
- `hand_processor/hand_data_left.npz`
- `hand_processor/hand_data_right.npz`
- `bbox_processor/bbox_data.npz`
- `segmentation_processor/masks_arm.npy`
- `depth.npy` placeholder, unused because `depth_for_overlay=false`
- camera intrinsics JSON
- `adapter_manifest.json`

After this point, action retargeting, smoothing, inpainting, robot rendering,
and compositing are original Phantom/Masquerade processors.

## Difference From Running The Unmodified EPIC Entry Point

The unmodified EPIC/Masquerade path begins with EPIC `hand_det.pkl`. EgoDex
does not have that pickle. Running the original `bbox` mode with `epic=true`
therefore cannot start unless an EPIC-shaped detection pickle is synthesized.

The current adapter bypasses only the EPIC detector/HaMeR front end by using
EgoDex HDF5 hand keypoints and arm transforms. This is the minimal compatibility
layer needed to feed EgoDex into the original downstream pipeline.

There are therefore two reproducibility levels:

- HDF5-keypoint baseline: uses EgoDex's provided hand/arm annotations as the
  dataset adapter, then runs original Phantom/Masquerade action, smoothing,
  E2FGVI, fixed-shoulders Kinova rendering, and robot compositing.
- Strict original frontend: additionally runs Phantom's original EPIC-style
  bbox, HaMeR hand2d, and SAM2 arm-segmentation frontend. This requires the
  HaMeR checkpoint and the licensed MANO_RIGHT.pkl file in
  `submodules/phantom-hamer/_DATA`.

Check local readiness with:

```bash
cd /mnt/project_rlinf/jlchen/code/phantom_reference
/mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.egodex_original.check_original_deps
```

Current dependency status:

- `hand_processor` imports successfully after installing the missing local
  frontend dependencies in the copied environment.
- SAM2, Detectron2, and E2FGVI weights are present.
- The HaMeR checkpoint and MANO_RIGHT.pkl are not present in the expected
  directory.
- The public HaMeR demo archive is 5.6GB; Google Drive was rate-limited and the
  fallback mirror estimated about 25 minutes to download. The partial download
  was removed because MANO_RIGHT.pkl is still required before the strict
  frontend can run faithfully.

## Difference From QwenRobot

QwenRobot's visual alignment stage differs from this baseline in several
important ways:

- It uses SAM3 text-prompt human-arm masks; this baseline uses EgoDex
  keypoint/arm-transform masks or optional SAM experiments.
- It uses ProPainter for clean background generation; this baseline uses the
  original Phantom E2FGVI processor.
- It searches robot base placement per morphology; this baseline uses the
  fixed Masquerade `shoulders` base.
- It renders and composites using scene depth reasoning; this baseline overlays
  robot/gripper mask pixels from MuJoCo and does not estimate scene depth.
- It retargets MANO keypoints with virtual finger `0.7 index + 0.3 middle`;
  this baseline uses Phantom's original `HandModel` grasp point/orientation.

## Current Verified Status

The current full original-repo baseline is:

`outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/video_overlay_Kinova3_shoulders.mp4`

Properties:

- 456x256
- 15 fps
- 1017 frames
- 67.8 seconds
- `training_data_shoulders.npz`: 1000 / 1017 valid frames
- invalid frames:
  `[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 995, 996, 997, 998, 999]`
- raw-hand preview:
  `outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/video_overlay_Kinova3_shoulders_rawhand.mp4`
- diagnostic montage:
  `outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/phantom_exact_epic_full_montage.jpg`
- summary:
  `outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/phantom_exact_epic_full_summary.json`

This baseline uses:

- original Masquerade fixed `Kinova3` + `Robotiq85` `shoulders` setup,
- original `ActionProcessor`,
- original `SmoothingProcessor`,
- original `HandInpaintProcessor` / E2FGVI,
- original `RobotInpaintProcessor`.

The EgoDex-specific part is still only the adapter that writes Phantom-shaped
hand keypoint, bbox, mask, camera, and video files from EgoDex HDF5/MP4.

The E2FGVI full output was checked by per-frame raw-vs-clean differences inside
the arm mask. On sampled frames `[0, 203, 406, 609, 812, 1016]`, masked-region
mean absolute RGB differences were approximately `[12.2, 17.0, 17.8, 20.2,
19.9, 16.1]`, confirming that the clean video is not just a copied raw
placeholder.

Recent smoke test for the adapter guardrail:

`outputs/phantom_egodex_original_baseline_check/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders.mp4`

Properties:

- 456x256
- 15 fps
- 30 frames
- 2.0 seconds
- `training_data_shoulders.npz`: 30 / 30 valid frames
- raw-hand preview:
  `outputs/phantom_egodex_original_baseline_check/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders_rawhand.mp4`
- diagnostic montage:
  `outputs/phantom_egodex_original_baseline_check/processed/egodex_phantom_original/0/phantom_original_repro_montage.jpg`

Observed result:

- The original fixed `Kinova3` shoulders renderer is reproducible and stable.
- The end-effector targets are close enough to the hands for this smoke segment,
  but the fixed robot roots and links do not cover EgoDex forearms.
- E2FGVI removes the broad arm mask only partially and leaves visible
  arm-shaped background residue.
- Therefore this baseline is useful as an original-repo control, but it is not
  a successful visual hand-to-robot replacement for EgoDex.

The raw-hand diagnostic is:

`outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/phantom_alignment_diagnostic_rawhand.jpg`

That diagnostic shows that the end-effector targets are close to the hand
finger proxy points. The visible failure is mainly visual: E2FGVI leaves
arm-shaped background smears, and the fixed-shoulders Kinova links do not trace
the exact human forearm silhouette.

## Fixed-Root Geometry Probe

After the full baseline, three 120-frame raw-hand geometry probes were run on
source frames starting at 1120. These probes keep the original Phantom
`ActionProcessor`, `SmoothingProcessor`, `RobotInpaintProcessor`, `Kinova3`,
`Robotiq85`, and bimanual `shoulders` setup. They only change the EgoDex
adapter's camera/extrinsics mapping:

- original EPIC shoulders extrinsics:
  `outputs/phantom_egodex_original_geom_original_1120/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders.mp4`
- root translation to EgoDex `Arm` anchors:
  `outputs/phantom_egodex_original_geom_root_arm_1120/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders.mp4`
- rigid arm-anchor alignment:
  `outputs/phantom_egodex_original_geom_arm_anchor_1120/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders.mp4`

All three runs produced 120 / 120 valid frames. The root-translation run aligns
the EgoDex upper-arm anchors to the fixed Phantom roots with about 1.35cm
residual per side. The rigid arm-anchor run fits the left-arm, left-hand,
right-arm, and right-hand anchors with errors of about 1.7cm, 1.8cm, 2.2cm,
and 3.7cm.

Visual comparison:

`outputs/phantom_egodex_original_geom_compare_1120.jpg`

Conclusion: the original fixed-root Kinova3 baseline is reproducible and the
grippers are close to the hands, but changing only the EgoDex adapter extrinsics
does not make the Kinova links trace or cover the EgoDex human forearms. The
remaining failure is a morphology / IK / visual-compositing mismatch, not a
single missing root translation.

## Next Step

Before adding QwenRobot components, the next practical target is to improve the
original-repo baseline's visual output without changing its action labels:

1. keep the fixed `Kinova3` shoulders robot,
2. keep Phantom action and smoothing,
3. either use a morphology / IK setup whose links actually follow the EgoDex
   forearm direction, or add a QwenRobot-style base and IK search constrained
   near the human shoulder side,
4. then replace the human-removal stage with stronger inpainting/background
   recovery and verify the final composited video.
