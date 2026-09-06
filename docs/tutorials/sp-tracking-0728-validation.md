---
title: SP-Tracking 0728 Validation
slug: /tutorials/sp-tracking-0728-validation
---

# SP-Tracking 0728 validation

Date: 2026-09-06. Source: `0728_baoshou_waist_dataclean_changedr`, iteration
22000. Runtime: root project, Python 3.10, MuJoCo 3.10.0, ONNX Runtime 1.23.2
CPU. Seed: `20260728`. Policy loop: 50 Hz; physics: 200 Hz. The source export
and adapted model identities are listed in the [deployment guide](sp-tracking-0728.md).

## Contract verification

- Source ONNX versus semantic-input ONNX: 20 random comparisons, maximum and
  mean action error both exactly 0. The model is ONNX IR 8 / opset 18.
- The 29 joint names, default positions, action scales, stiffness, and damping
  match the source deployment `sim2real/config/g1/tracking_spv5_2.yaml` exactly.
- Training analytic FK versus deploy MuJoCo Jacobians: 100 random q/dq/gyro
  states, maximum absolute feature error `1.9073486e-6`, mean `5.1618059e-8`.
  This uses the training `g1_tracking_bfm/g1.xml` and the source helper API.
- 26 focused tests pass, including history ordering and shared updates,
  torque propagation through Robot I/O, and action clipping before scaling
  and before the returned action is stored for the next observation.
- Source deployment clips actions to ±10. Both runtime paths now honor the
  optional `clip_actions` YAML field; configs without it retain their existing
  behavior.
- Both launcher modes pass argument inspection with `--dry-run`; local NPZ
  and Pico/ZMQ configurations are covered. CPU ONNX inference returns 29 actions.
  Syntax checks pass for all changed Python runtime and adapter files.

Reproduce the source FK check with the training checkout available:

```bash
uv run --no-sync python scripts/validate_sp_tracking_kinematics.py \
  --source-repo ../SP_Tracking --trials 100 --seed 20260728
uv run --no-sync --with pytest pytest -q \
  tests/test_sp_tracking_observations.py tests/test_robot_io.py
```

## Integrated sim2sim

Both runs initialize the robot from motion frame 0 and keep the policy active
during the initial two-second pause. They use the shared simulation robot and
its normal contact/dynamics settings, with no training randomization.

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --headless --run-once \
  --root-trajectory-output outputs/sp_tracking_0728/walk_verified_root.npz

uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --motion-path datasets/root90/motions/backward__00__FairySteps.npz \
  --headless --max-runtime-s 20 \
  --root-trajectory-output outputs/sp_tracking_0728/backward_verified_hold_root.npz
```

The default walking NPZ contains 7,840 qpos frames; the existing motion loader
resamples it to 13,066 policy frames at 50 Hz. The backward clip contains 401
policy frames. Each saved NPZ records the exact policy/motion paths, seed,
robot/reference trajectories, start/end positions, and relative final error.
Root displacement errors are computed in each trajectory's initial root frame.

| Metric | `walk1_subject1` | `backward__00__FairySteps` |
| --- | ---: | ---: |
| Final reference frame reached | 13065 / 13065 | 400 / 400 |
| Total simulation time | 263.32 s | 20.00 s |
| Final-frame hold | Exit at end | 10.015 s |
| Minimum robot root height | 0.626 m | 0.775 m |
| Robot relative final displacement (x, y, z), m | (1.118, 0.569, 0.046) | (−0.953, 0.127, −0.024) |
| Reference relative final displacement (x, y, z), m | (0.465, −1.679, 0.019) | (−1.522, 0.153, −0.122) |
| Final root displacement error, 3D | 2.3417 m | 0.5786 m |
| Final root displacement error, XY | 2.3416 m | 0.5701 m |

The backward metrics include the final-frame hold interval. Both trajectories
remain finite and reach their final reference frame; neither run shows a fall.

These are deployment smoke tests, not a benchmark-wide accuracy claim. The
long walking clip shows appreciable accumulated position error even though it
completes without a fall. No G1 hardware connection, motor motion, GPU backend,
or onboard timing was tested. Real deployment commands are prepared but do
not constitute hardware validation.
