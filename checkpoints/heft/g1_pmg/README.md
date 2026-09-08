# HEFT G1 PMG

Chinese: [README_zh.md](README_zh.md).

This packages the default G1 PICO tracking policy from the local
`/home/xingyiwang/workspace/motion_tracking` repository. Its released model is
`sim2real/config/g1/ckpts/G1_PMG/policy.onnx` on branch `sim2real`, commit
`0d5ba31e33397f3543d350d98b637e26d92f470a`. The current training branch is
`main`, commit `731c3c41a82183153e25ea302612290e88346177`; its G1 student
observation definitions were also checked. This package uses the released PMG
model. A new training run or the Compliance model requires its own export/config.

`policy.onnx` contains the original observation normalization, student adaptor,
and actor with embedded weights. The four semantic inputs are concatenated
inside the graph. The only output is the deterministic 29-joint actor mean
named `action`, numerically identical to the source runtime's action alias.
`policy.json` records source file hashes, provenance, and export parity results.

## Observation and action contract

| Input | Shape | Contents, in source order |
| --- | --- | --- |
| `context` | `[1, 1]` | Boot countdown, first reset sample 24/25 |
| `motion_command` | `[1, 114]` | Reference root displacements, then relative root rotations |
| `target_motion` | `[1, 806]` | Absolute reference joints, reference minus current joints, root heights, reference gravity |
| `proprioception` | `[1, 808]` | Gyro, gravity, absolute joints, joint velocities, previous raw actions |

Reference offsets are `[0,1,2,3,4,5,6,-1,-2,-4,-8,-12,-16]` at 50 Hz.
All four proprioception histories select `[0,1,2,3,4,8,12,16,20]`, newest first;
previous actions contain eight consecutive samples. Histories start at zero.
Semantic inputs share one stateful core and advance once per inference step.
Reset discards the old action left in the generic runtime.

Root displacements use the current reference's full orientation. Relative
rotation uses the actual robot orientation and serializes the first two matrix
columns, column first. Quaternions are wxyz and normalized like the source
SciPy implementation. Joint positions do not subtract the standing pose.
Body and joint indices refresh when the PICO stream supplies its actual names.

The YAML preserves the source controller's interleaved 29-joint action order,
default pose, and PD gains. Actions clip to ±10 before applying
`target = default_joint_pos + action_scale * action`; scales are 1.0 for arms
and 0.5 elsewhere. All reference/action mapping uses names, including conversion
to the MuJoCo/Unitree joint order.

## Run PICO Sim2Sim

From the repository root, use three terminals. Start the publisher:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py --robot g1
```

Start MuJoCo:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python sim2real/sim_env/base_sim.py --robot g1
```

Start the policy after stopping any previous tracker using command port 5591:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/heft/g1_pmg/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --motion-backend zmq \
  --motion-zmq-connect tcp://127.0.0.1:28701 \
  --controller pico \
  --pico-zmq-connect tcp://127.0.0.1:5592 \
  --rl-rate 50
```

With the publisher initially paused, press `A` for the init pose, then `A+B`
for policy control, then `X` while standing naturally to begin live following.
The existing publisher handles controller buttons and body streaming. If it
prints `Waiting for XR body data from PICO`, enable body streaming in the PICO
XRoboToolkit connection first; loading this model does not start device streaming.

This integration uses this repository's PICO workflow: pause returns to its
default standing reference, and resume aligns horizontal position and heading.
The upstream VR pipeline's two-second start blend and hold-last-pose behavior
are not reproduced. Its request/reply transport is also different; do not run
the upstream VR server on this publisher's port. Here the six future frames
need 120 ms lookahead plus the configured 40 ms tolerance, totaling 160 ms of
buffering. Live device behavior has not yet been measured.

CPU ONNX Runtime was verified locally. This graph retains upstream IR 10 /
opset 20; an onboard ORT 1.16 environment needs the compatibility conversion
described in the adaptation skill before GPU use.

## Validation on 2026-09-08

- 64 structured nonzero input comparisons: converted versus original ONNX
  maximum absolute action error **0.0**; repeated inference was deterministic.
- 12 HEFT tests passed, including direct source-observation comparison across
  6 × 53 control steps (tolerance 2e-6), history/reset, and live name reordering.
  Together with the motion-buffer regression suite, **48 tests passed**.
- Synthetic 30 Hz PICO JSON through the actual 50 Hz `Tracking.step` path:
  correct four input shapes, finite actions, command order/scale error **0.0**.
  CPU single-thread step mean **1.744 ms**, max **2.228 ms** over 180 samples.
  Robot I/O was replaced with a test double for this audit.

Integrated MuJoCo tests placed the robot at motion frame zero, ran active policy
control during a two-second initial pause, played the clip, then held its final
frame. Seed `20260908`, policy 50 Hz, physics 200 Hz, `onnx-cpu`:

| Reference | Sim duration | Minimum pelvis Z | Maximum pelvis tilt | Final horizontal displacement error | Final hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source default standing pose | 12 s | 0.774 m | 2.47° | 0.0010 m | 2.02 s |
| Current PICO paused pose | 10 s | 0.782 m | 1.06° | 0.0006 m | 6.02 s |
| Source `walk1_subject1`, first 30 s | 34 s | 0.735 m | 8.97° | 0.1655 m | 2.02 s |
| `dance1_subject2_0_3945`, first 30 s | 34 s | 0.565 m | 23.57° | 0.9800 m | 2.02 s |

All four runs completed without falling; the dance result still has substantial
position drift. Displacement errors compare robot/reference final displacement
in their respective initial root frames, not joint tracking accuracy. These
results establish the tested integration behavior, not live PICO stability.

Local audit commands, logs, JSON metrics and root trajectories are in
`outputs/heft_adaptation_audit/`; repeat the prepared fixtures with
`uv run --no-sync python outputs/heft_adaptation_audit/run_validation.py`.
The source walk NPZ uses xyzw root rotations; its audit fixture converts them to
wxyz qpos, preserving the named joint order and native 50 Hz frame rate.

## Reproduce the model package

Extract the deployed branch without changing the training checkout:

```bash
mkdir -p external/motion_tracking_sim2real
git -C /home/xingyiwang/workspace/motion_tracking archive \
  0d5ba31e33397f3543d350d98b637e26d92f470a sim2real README.md .gitattributes \
  | tar -x -C external/motion_tracking_sim2real
uv run --no-sync python scripts/export_heft_policy.py \
  --source-root external/motion_tracking_sim2real
uv run --no-sync --with pytest python -m pytest \
  tests/test_heft_observations.py tests/test_motion_buffer.py -q
```

Direct source parity tests require that extracted source. Independent history,
layout, and reset regressions still run without it.
