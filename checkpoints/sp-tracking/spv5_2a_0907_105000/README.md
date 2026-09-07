# SP-Tracking SPV5-2A / 105000

Source run: `2026-09-06_02-31-13_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct`
Source iteration: `105000`

`policy.onnx` preserves the source actor and adds four named semantic inputs.
The training `.pt` file is provenance-only and is not copied into this runtime artifact.

Deploy ONNX SHA-256:
`9a4f37c1e6226a8749da2ce931bd8dfc796ff29746b67bea3f4bcd8a912e8aaf`.
Size: 64,830,945 bytes (61.83 MiB). Use the generic runtime below;
`scripts/run_sp_tracking_0728.py` intentionally accepts only the 0728 model.

## Sim2sim

Run from the repository root with `inference-cpu` installed. Prepare a local
G1 any4hdmi motion and the offline robot asset cache first; see the
[0728 setup guide](../../../docs/tutorials/sp-tracking-0728.md) for asset preparation.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/sim_env/integrated_sim2sim.py \
  --policy-config checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --env-dt 0.02 --initial-pause-s 2
```

Add `--headless --run-once` for a non-visual run that exits at the final frame.
Without `--run-once`, the simulator continues holding the final reference.

## G1 sim2real

Install the root `inference-cpu` and `robot-g1` extras and prepare the local
motion and asset cache; see [Robot I/O](../../../docs/robot_io.md).
Persist `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` exports in the
robot user's `~/.bashrc`. Replace `eth0` with the verified motor-network interface.
The following command opens real robot I/O; start only after the normal G1
safety checks, with support and an emergency-stop operator ready.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/rl_policy/tracking.py \
  --robot g1 --robot-io inline --robot-interface eth0 --controller joystick \
  --policy-config checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml \
  --motion-backend npz --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --rl-rate 50
```

Joystick: `A` default pose, `R1` policy mode, `B` pause/resume reference,
`R2` zero torque. Do not use zero torque as a substitute for physical support.

## Validation (2026-09-07)

ONNX checker, metadata/hash checks, and CPU single-step inference passed.
A headless integrated sim2sim run completed all 401 frames of
`datasets/root90/motions/backward__00__FairySteps.npz`, with a two-second
initial pause and seed `20260728`. The local trajectory is
`outputs/sp_tracking_105000/upload_smoke_root.npz` (not committed).
The existing export metadata records 20 source-versus-adapted comparisons with
zero action error; those source comparisons were not rerun for this upload.
This is a smoke test, not a general tracking-quality or real-G1 safety validation.
No real-robot execution was performed.
