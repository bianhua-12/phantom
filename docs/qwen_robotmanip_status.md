# Qwen RobotManip Reproduction Status

This document records the current state of the Qwen RobotManip reproduction work
inside this Phantom/Masquerade fork. It is the current engineering view, not a
claim that the visual result is solved.

## Current Mainline

The default forward path is the newer Qwen-style pipeline in `phantom/qwenrobot/`:

```bash
python -m phantom.qwenrobot.egodex_retarget
python -m phantom.qwenrobot.search_base
python -m phantom.qwenrobot.render_robot --placement-mode world_base
```

This path keeps Qwen-style hand retargeting and explicit MuJoCo base/IK
evaluation closer to the intended reproduction than the older Phantom shoulders
patches.

The original Phantom/Masquerade processors remain the baseline and regression
control. The Phantom shoulders base-offset experiments are historical probes:
they are useful evidence, but should not be treated as the final base-search
design.

## Completed Work

- Gripper width handling was changed from cup-demo binary open/close behavior to
  continuous thumb-to-virtual-finger width. `Robotiq85` remains the mainline
  gripper; `Robotiq140` mapping support exists for comparison.
- Qwen canonical grasp frame was introduced with `x=approach`, `y=normal`,
  `z=jaw axis`. Phantom/Robosuite adapter code compensates for the downstream
  fixed `+135deg z` tool rotation.
- Initial base search probes exist:
  - `phantom/qwenrobot/search_phantom_base_offsets.py` searches Phantom
    shoulders offsets, yaw, and tool roll through the Robosuite controller.
  - `phantom/qwenrobot/search_base.py` searches explicit `base_xyz_yaw` for the
    newer Qwen/MuJoCo path and writes `03_manifest.json`.
- Robot overlay video output is now guarded against all-black results. Failed
  tracking frames keep the background frame instead of writing black frames.
- ProPainter is wired for clean-background experiments. It can remove hands, but
  the current jigsaw result has visible artifacts and should be treated as a
  partial background, not final data quality.

## Known Failures

- The biggest current failure is base-search parameterization. Previous Phantom
  shoulders searches explored offsets around the default shoulders base, which is
  near the camera/person. That can produce `valid` tracking while placing the
  robot visually against the face/camera, so it is not equivalent to Qwen's base
  search.
- Controller and base-search frames are not consistently synchronized in the
  Phantom shoulders path. The robot model base may be moved, while action
  transforms and tracking-error evaluation still largely follow the original
  shoulders frame.
- Tracking validity alone is not a useful visual objective. Existing searches
  can report all keyframes valid while producing a visually unusable overlay.
  Robot mask area and bottom-area metrics are only diagnostic patches, not a
  replacement for correct base placement.
- Jaw-axis tracking remains unresolved. The fixed adapter rotation is plausible,
  but actual OSC/IK results can still show jaw-axis errors above 100 degrees,
  indicating weak orientation tracking or an unreachable pose under the current
  morphology/base combination.
- The jigsaw ProPainter background is only marginally usable. It removes hands,
  but leaves notable smearing and lower-left artifacts.

## Current Priority

The older baseline reports recommended cleaning the background before continuing
base search. That was correct for the original fixed-root Phantom control path,
but it is not the current Qwen reproduction priority.

The current priority order is:

1. Redesign Qwen-style base search around an explicit real `T_world_base` /
   `T_base_world`, not offsets around Phantom shoulders.
2. Make that base transform affect robot model placement, controller target
   transforms, and error evaluation through one shared frame contract.
3. Rebuild IK feasibility around representative keyframes with position and
   jaw-axis/orientation terms, rather than treating OSC tracking as the only
   feasibility signal.
4. Add hard visual constraints: robot base and large arm segments must not enter
   the first-person image in implausible regions.
5. Rerender jigsaw only after the base/frame/IK contract is coherent.
6. If `Kinova3 + Robotiq85` still fails, compare alternative morphologies such
   as longer arms, smaller grippers, `Robotiq140`, Panda, or ALOHA-style arms.

## Non-Goals For This Documentation Pass

- No Python API or command-line interface changes.
- No action schema, NPZ layout, or training-data format changes.
- No rerendering, IK experiment, ProPainter rerun, or output deletion.
- No claim that any existing `valid_ratio=1.0` output is visually correct.

## Suggested Next Interface

The next implementation should introduce a single base-frame contract used by
retargeting, search, IK, render, and metrics. The minimum concept is:

- `T_world_base`: candidate robot base pose in the scene/trajectory world frame.
- `T_base_world`: inverse transform used to express Qwen targets in robot-local
  coordinates.
- A shared evaluator that reports position error, jaw-axis error, rotation error,
  robot image area, base visibility, and keyframe validity from the same
  candidate transform.
