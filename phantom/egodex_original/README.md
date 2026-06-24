# EgoDex With The Original Phantom/Masquerade Pipeline

This folder is the EgoDex reproduction path that keeps the original
Phantom/Masquerade robotization stages fixed:

- Robot: `Kinova3` + `Robotiq85`
- Setup: bimanual `shoulders`
- Retargeting: original `ActionProcessor`
- Smoothing: original `SmoothingProcessor`
- Human removal: original `HandInpaintProcessor` / E2FGVI
- Robot rendering and compositing: original `RobotInpaintProcessor`

The only adapter-specific work is converting an EgoDex `.hdf5`/`.mp4` pair into
the file layout expected by those processors.

This is a control baseline, not the final QwenRobot-style visual alignment. On
the current EgoDex demo, the original fixed `shoulders` roots do not cover the
human forearms well, even though the end-effectors track the hands reasonably.
See `IMPLEMENTATION_REPORT.md` for the latest smoke-test evidence.

## Smoke Test

```bash
cd /mnt/project_rlinf/jlchen/code/phantom_reference
CUDA_VISIBLE_DEVICES=0 /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.egodex_original.run \
  --output-root outputs/phantom_egodex_repro_original_smoke \
  --demo-index 0 \
  --demo-name egodex_phantom_original \
  --demo-num 0 \
  --frame-stride 2 \
  --max-frames 90 \
  --overwrite
```

Verified output:

- `outputs/phantom_egodex_repro_original_smoke/processed/egodex_phantom_original/0/video_overlay_Kinova3_shoulders.mp4`
- 456x256, 15 fps, 90 frames, 6.0 seconds
- `training_data_shoulders.npz`: 90 / 90 valid frames

## Full Demo Command

```bash
cd /mnt/project_rlinf/jlchen/code/phantom_reference
CUDA_VISIBLE_DEVICES=0 /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.egodex_original.run \
  --output-root outputs/phantom_egodex_repro_original_full \
  --demo-index 0 \
  --demo-name egodex_phantom_original \
  --demo-num 0 \
  --frame-stride 2 \
  --max-frames full \
  --overwrite
```

An equivalent full-length baseline already exists at:

- `outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/video_overlay_Kinova3_shoulders.mp4`
- 456x256, 15 fps, 1017 frames, 67.8 seconds
- `training_data_shoulders.npz`: 1000 / 1017 valid frames

## Alignment Diagnostic

The main geometry check is the raw-hand overlay: render the original Phantom
fixed-shoulder Kinova3 trajectory directly on top of the unedited EgoDex RGB
video. The gripper/action targets should land on the visible hand before any
inpainting quality work is attempted.

```bash
cd /mnt/project_rlinf/jlchen/code/phantom_reference
CUDA_VISIBLE_DEVICES=0 /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.egodex_original.diagnose_alignment \
  --processed-demo-dir outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0 \
  --frames 0 120 240 360 520 680 840 1000 \
  --output-prefix phantom_alignment_diagnostic_rawhand
```

Verified on demo 0:

- `phantom_alignment_diagnostic_rawhand.jpg` overlays EgoDex hand keypoints and
  projected Phantom action targets, plus the raw-hand robot overlay.
- Mean detected-frame action-target-to-hand-tip-proxy error:
  left 2.64 px, right 2.82 px.
- The complete raw-hand overlay video is
  `video_overlay_Kinova3_shoulders_rawhand.mp4`, 456x256, 15 fps, 1017 frames,
  67.8 seconds.

This diagnostic uses the original Masquerade-style fixed `Kinova3` shoulders
setup. It does not include QwenRobot base search, SAM3, ProPainter, Depth
Anything, or multi-morphology rendering.

## Fixed-Root Adapter Probe

To check whether the poor EgoDex coverage is only a root-placement issue, three
120-frame raw-hand probes were run from source frame 1120:

- `outputs/phantom_egodex_original_geom_original_1120/.../video_overlay_Kinova3_shoulders.mp4`
- `outputs/phantom_egodex_original_geom_root_arm_1120/.../video_overlay_Kinova3_shoulders.mp4`
- `outputs/phantom_egodex_original_geom_arm_anchor_1120/.../video_overlay_Kinova3_shoulders.mp4`

Comparison image:

- `outputs/phantom_egodex_original_geom_compare_1120.jpg`

The original repo baseline is reproducible, but these probes show that adapter
extrinsics alone do not make the fixed Kinova3 links cover EgoDex forearms. The
next correction has to change the morphology / IK / base-search stage rather
than only translating the fixed root.

## SAM3.1 + ProPainter Visual Swap Experiment

This keeps the original Phantom/Masquerade action, smoothing, fixed `shoulders`
base, `Kinova3` robot, and `RobotInpaintProcessor`. Only the human-removal
background is swapped to SAM3.1 masks plus ProPainter.

```bash
cd /mnt/project_rlinf/jlchen/code/phantom_reference
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/project_rlinf/jlchen/code/phantom_reference \
  /mnt/project_rlinf/jlchen/envs/sam3_py312/bin/python \
  -m phantom.egodex_original.sam31_segment \
  --processed-demo-dir outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0 \
  --output-name masks_arm_sam31_d5.npy \
  --close-kernel 5 \
  --post-dilation 5

CUDA_VISIBLE_DEVICES=0 /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.qwenrobot.propainter_inpaint_demo \
  --processed-demo-dir outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0 \
  --mask-path outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/segmentation_processor/masks_arm_sam31_d5.npy \
  --output-video outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/inpaint_processor/video_human_inpaint_propainter_sam31_d5.mkv \
  --work-dir outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/propainter_processor/sam31_d5_full \
  --fps 15 \
  --mask-dilation 2 \
  --subvideo-length 80 \
  --neighbor-length 10 \
  --ref-stride 10 \
  --raft-iter 20 \
  --fp16

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.qwenrobot.rerender_robot_overlay \
  --processed-demo-dir outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0 \
  --inpaint-video outputs/phantom_egodex_exact_epic_full/processed/egodex_phantom/0/inpaint_processor/video_human_inpaint_propainter_sam31_d5.mkv \
  --robot Kinova3 \
  --gripper Robotiq85 \
  --suffix sam31_d5 \
  --overwrite \
  --export-mp4
```

Verified output:

- `video_overlay_Kinova3_shoulders_sam31_d5.mp4`
- `video_overlay_Kinova3_shoulders_sam31_d5_preview_filled.mp4`
- `phantom_sam31_propainter_phantom_compare.jpg`
- 456x256, 15 fps, 1017 frames, 67.8 seconds
- `training_data_shoulders_sam31_d5.npz`: 1000 / 1017 valid frames
- Invalid frames:
  `[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 995, 996, 997, 998, 999]`

The SAM3.1 mask covers the visible hands and forearms well. The current
remaining artifact is ProPainter background recovery at 256p: dark arm-shaped
smears remain after inpainting. The original Phantom robot renderer overlays
only the rendered robot/gripper mask, so it will not fill every pixel removed by
the broader human-arm mask.

## SAM2 Mask Refinement

SAM2 is wired as an optional mask refinement pass:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/project_rlinf/jlchen/envs/qwen_visual_align/bin/python \
  -m phantom.egodex_original.sam2_refine_masks \
  --processed-demo-dir outputs/phantom_egodex_repro_original_smoke/processed/egodex_phantom_original/0 \
  --max-anchors 3 \
  --dilation 21 \
  --close-kernel 13 \
  --post-dilation 5
```

On the 90-frame smoke test, SAM2 refinement does not solve the visible artifact:
the robot still aligns with the hand region, but E2FGVI leaves background smears
when large hand/forearm regions are removed. The next meaningful improvement is
better video inpainting/background recovery, not base-search tuning.
