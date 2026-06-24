# EgoDex on Phantom/Masquerade Baseline

This baseline intentionally follows the original Phantom/Masquerade code path before
adding QwenRobot-specific pieces.

## Original Phantom/Masquerade Path

- Robot: `Kinova3` with `Robotiq85`, bimanual `shoulders` setup.
- Base placement: fixed hard-coded shoulder roots in `PhantomBimanual._load_model`.
- Camera/action frame: Epic/Masquerade camera intrinsics and
  `camera_extrinsics_ego_bimanual_shoulders.json`; `TwinBimanualRobot` applies
  `BASE_T_1` when `epic=True`.
- Action retargeting: `ActionProcessor` converts 3D hand keypoints to robot frame,
  fits the repo `HandModel`, and outputs end-effector pose plus gripper width.
- Smoothing: repo `SmoothingProcessor` uses GP position/width smoothing and
  Gaussian-weighted SLERP for rotations.
- Visual edit: repo `HandInpaintProcessor` uses E2FGVI, then
  `RobotInpaintProcessor` renders and overlays the robot through MuJoCo masks.

## EgoDex Adapter Scope

The adapter only supplies files expected by the original processors:

- `video_L.mp4`, `video_R.mp4`, `video_rgb_imgs.mkv`
- `hand_processor/hand_data_{left,right}.npz`
- `segmentation_processor/masks_arm.npy`
- camera intrinsics JSON scaled to the resized EgoDex frames
- optional `inpaint_processor/video_human_inpaint.mkv`

For this baseline, the adapter keeps `retarget=phantom`,
`epic=True`, `bimanual_setup=shoulders`, and the original Epic shoulder
extrinsics. It does not use QwenRobot action retargeting or base search.

## Current QwenRobot Differences Not Yet Used

- QwenRobot defines gripper pose from thumb, index, and middle fingertips directly.
- QwenRobot uses Savitzky-Golay position/width smoothing and Gaussian SLERP.
- QwenRobot searches robot base placement by IK feasibility around the trajectory
  centroid.
- QwenRobot uses SAM3 masks, ProPainter inpainting, Depth Anything metric depth,
  and depth-based compositing.
- QwenRobot renders 15 robot morphologies; this baseline uses only the original
  Kinova3 shoulders morphology.

## Verification Artifacts

For every episode, the important outputs are:

- `video_overlay_Kinova3_shoulders.mp4`: final robotized video after hand inpaint.
- `video_overlay_Kinova3_shoulders_rawhand.mp4`: the same robot trajectory on the
  original RGB video. This is the main coverage diagnostic; the gripper should
  land on the visible hand before any QwenRobot replacement is attempted.
- `training_data_shoulders.npz`: frame-level valid flags and action labels.

## Demo-0 Full-Video Check

Processed demo:
`outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0`.

Original Phantom/Masquerade baseline:

- `video_overlay_Kinova3_shoulders.mp4`
- 456x256, 15 fps, 1017 frames, 67.8 seconds.
- `training_data_shoulders.npz`: 1000 / 1017 valid frames.
- Invalid frames:
  `[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 995, 996, 997, 998, 999]`.

Raw-hand coverage diagnostic:

- `video_overlay_Kinova3_shoulders_rawhand.mp4`
- Visual inspection on representative frames shows the fixed-shoulder Kinova3
  arms enter from the human side and the grippers generally land near the visible
  hands/fingertips. This indicates the current largest visible artifact is the
  hand-removal/background step rather than a total robot-base-side failure.
- Quantitative diagnostic from
  `phantom_alignment_diagnostic_rawhand.json`: projecting Phantom smoothed action
  targets back into the EgoDex image gives mean detected-frame error of 2.64 px
  for the left hand and 2.82 px for the right hand against the fingertip proxy.
  This is the main evidence that the fixed-root Phantom/Masquerade action and
  camera frames are aligned for demo 0.

ProPainter swap experiment, not the baseline:

- `inpaint_processor/video_human_inpaint_propainter.mkv`
- `video_overlay_Kinova3_shoulders_propainter.mp4`
- 456x256, 15 fps, 1017 frames, 67.8 seconds.
- `training_data_shoulders_propainter.npz`: 1000 / 1017 valid frames.
- Invalid frames:
  `[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 754, 996, 997, 998, 999]`.
- ProPainter is wired to the pipeline, but this run still leaves large background
  smears because the current arm masks are broad and synthetic. It should not be
  treated as a solved QwenRobot visual-alignment implementation.

SAM3.1 + ProPainter + original Phantom renderer experiment, still not a solved
QwenRobot implementation:

- SAM3.1 mask:
  `segmentation_processor/masks_arm_sam31_d5.npy`
- Mask montage:
  `segmentation_processor/masks_arm_sam31_d5_montage.jpg`
- ProPainter clean background:
  `inpaint_processor/video_human_inpaint_propainter_sam31_d5.mkv`
- Phantom original fixed-root Kinova3/Robotiq85 overlay:
  `video_overlay_Kinova3_shoulders_sam31_d5.mp4`
- Viewing-only filled preview with invalid black frames copied from nearest valid
  frame:
  `video_overlay_Kinova3_shoulders_sam31_d5_preview_filled.mp4`
- 456x256, 15 fps, 1017 frames, 67.8 seconds.
- `training_data_shoulders_sam31_d5.npz`: 1000 / 1017 valid frames.
- Invalid frames:
  `[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 995, 996, 997, 998, 999]`.
- Comparison montage:
  `phantom_sam31_propainter_phantom_compare.jpg`
- Visual diagnosis: SAM3.1 covers both visible human hands and forearms well.
  The raw-hand robot overlay shows that the fixed-root Phantom grippers are on
  the human-hand side and generally land on the visible manipulation region.
  The remaining artifact is mainly ProPainter background recovery at 256p:
  dark arm-shaped smears remain after inpainting. The Phantom renderer then
  overlays only the rendered robot body/gripper mask, so it does not fill every
  pixel removed by the human-arm mask.

Diagnostic montage:

- `outputs/diagnostics/current_mask_inpaint/phantom_exact_full_comparison.png`

Current conclusion:

1. Keep the original Phantom fixed-root/Kinova3 path as the reproducibility
   baseline.
2. Do not tune base search until the fixed-root path is clean.
3. Next priority is better background recovery for the SAM3.1 arm/shadow mask
   while preserving the raw-hand alignment diagnostic.
4. Only after that replace the action/base/depth parts with QwenRobot-specific
   components.
