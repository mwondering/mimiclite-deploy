---
title: Deploy SP-Tracking 0728
slug: /tutorials/sp-tracking-0728
---

# Deploy SP-Tracking 0728

This profile runs `0728_baoshou_waist_dataclean_changedr/checkpoint_22000.pt`
through its matching complete ONNX export. Both sim2sim and G1 deployment use
`checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml`.
The dedicated launcher verifies the source identity and adapted ONNX SHA-256
before starting the runtime. The older `checkpoints/sp-tracking/spv5_2/` local
artifact contains the same weights; the explicit directory identifies this run.

The repository includes the deploy ONNX, JSON metadata, YAML, and kinematics XML.
The training `.pt` and motion datasets are not included. No source training
checkout is required to run the prepared policy.

## Sim2sim

Use the root Python environment with the CPU inference extra. Put a G1 motion
at `datasets/lafan40/motions/walk1_subject1.npz`, or pass another local any4hdmi
G1 NPZ with `--motion-path`. The default motion can be prepared on an online PC:

```bash
uv run --no-sync hf download elijahgalahad/any4hdmi-g1-lafan \
  motions/walk1_subject1.npz --local-dir datasets/lafan40
```

Run from the repository root:

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim
```

This opens the integrated MuJoCo viewer, initializes the robot from motion
frame 0, runs the policy during a two-second initial pause, and then plays the
motion. At the end, the policy continues holding the final reference frame.
The simulation step is 5 ms; the policy and reference step is 20 ms (50 Hz).
The launcher sets `SIM2REAL_ORT_NUM_THREADS=1` unless already configured.

For a headless run with a saved root trajectory:

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --headless --run-once \
  --root-trajectory-output outputs/sp_tracking_0728/root.npz
```

`--run-once` exits at the final reference frame. Omit it to test continued
final-frame holding; `--max-runtime-s` bounds the simulated duration.
Use `sim2sim --help` for all runtime options.

## G1 sim2real

Use the root project environment on G1, with `inference-cpu` and `robot-g1`
installed as described in [Robot I/O](../robot_io.md). Copy the repository,
the policy artifact, the chosen motion, and any required offline robot-asset
cache to G1. Persist these lines in the robot user's `~/.bashrc`:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

First inspect settings without opening DDS or sending robot commands:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 --dry-run
```

For local NPZ playback on the robot:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --record --record-output outputs/sp_tracking_0728/real.npz
```

Replace `eth0` with the G1 motor-network interface. This selects inline G1
robot I/O and the Unitree joystick: `A` enters the default pose, `R1` enables
the policy, `B` resumes/pauses the reference, and `R2` enters zero-torque mode.
For terminal control, use `--controller keyboard`: `i`, `]`, Space, and `o`
perform the corresponding operations.

For an existing G1 ZMQ motion publisher, including Pico retargeting:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 --motion-backend zmq --controller pico
```

Start the publisher using the [Pico teleoperation tutorial](pico-teleoperation.md).
The launcher sets both Hugging Face offline variables before importing the
runtime. CPU inference is the portable default; `--inference-backend onnx-gpu`
or `tensorrt` uses the repository's existing backends when installed and verified
on the target machine. `--dry-run` does not validate DDS connectivity or hardware
inference speed. No real-robot motion test was performed for this change.

## Observation and action contract

The ONNX preserves the complete actor, reference encoder, estimator, and learned
normalizers. An internal Concat replaces the original flat input with these
semantic inputs, in source order:

| Input | Shape | Meaning |
| --- | --- | --- |
| `robot_root_quat` | `1 × 4` | Pelvis quaternion, wxyz. |
| `estimator_history` | `1 × 6100` | 50 samples, term-major and oldest first: joint position relative to default, joint velocity, gravity, body gyro, previous action, latest measured joint torque. |
| `reference_encoder_input` | `1 × 1900` | 50 reference frames at offsets −42 through +7: world root position, column-major 6D rotation, joint position. |
| `robot_key_body` | `1 × 195` | 13 semantic body points from measured q/dq/gyro and training-matched FK. |

All input groups share one history update per policy step. Deployment disables
training observation noise. The output is 29 joint actions in the source G1
MuJoCo order, clipped to ±10 and transformed using the source action scales,
default pose, and PD gains in the YAML. The future-reference requirement is
0.14 s. Real-robot input uses IMU, joint encoders, and latest motor torque;
measured global root position/velocity and external motion capture are not
required by this actor.

The model uses ONNX IR 8 / opset 18. The source export SHA-256 is
`65283dfa5f48c51dc28d2873852ecf379b04c2d0a97a23a4bb5b924a486684f2`;
the adapted model SHA-256 is
`c88d39cfe823c801dd7ada9d029360744dc692af900ce955824d6904cb978ef4`.

## Rebuild the deployment artifact

```bash
uv run --no-sync python scripts/adapt_sp_tracking_spv5_2.py \
  --checkpoint-dir ../motion_tracking_sim2real_self/ckpts/0728_baoshou_waist_dataclean_changedr \
  --iteration 22000 \
  --output-dir checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr \
  --equivalence-trials 20
```

Twenty CPU source-versus-adapted ONNX comparisons passed with exactly zero
action error. Validation commands and simulation metrics are recorded in
[the experiment log](sp-tracking-0728-validation.md).
