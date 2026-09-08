---
title: Pico Teleoperation
sidebar_position: 2
---

This tutorial connects PICO / XR full-body tracking to **SONIC release SMPL** and uses the root project tracking policy to control G1. The model is the default SONIC release from `GR00T-WholeBodyControl`, preserving the SMPL mode used by its PICO full-body teleoperation.

The complete models and `human_joints_info.pkl` skeleton are included. See the
[artifact notes](https://github.com/mwondering/mimiclite-deploy/blob/main/checkpoints/sonic/release/README.md)
for provenance and validation. Controls use this repository's `A`, `A+B`, and `X`; these commands enter full-body SMPL tracking without starting the source project's planner or VR three-point mode.

## 1. Start the Pico retarget publisher

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --robot g1 \
  --actual-human-height 1.76 \
  --publish-hz 30 \
  --publish-smpl
```

Set `--actual-human-height` to your height in meters. `--publish-smpl` is required and sends the SMPL reference on port `28702`.

The publisher's mjviser view shows the parallel GMR robot reference and can help inspect the tracking connection. SONIC SMPL consumes human skeleton and wrist references, which are a different representation. Enable streaming of both PICO full-body and controller data.

## 2. Choose the execution backend

### Sim2Sim

Start the MuJoCo execution process:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run sim2real/sim_env/base_sim.py --robot g1
```

In another terminal, start the tracking policy against the live motion stream:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sonic/release/smpl/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --motion-backend smpl_zmq \
  --motion-zmq-connect tcp://127.0.0.1:28702 \
  --controller pico \
  --pico-zmq-connect tcp://127.0.0.1:5592 \
  --rl-rate 50
```

### Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). The Pico-specific policy flags stay the same:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sonic/release/smpl/policy.yaml \
  --motion-backend smpl_zmq \
  --motion-zmq-connect tcp://127.0.0.1:28702 \
  --controller pico \
  --pico-zmq-connect tcp://127.0.0.1:5592 \
  --rl-rate 50
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Pico Controls

- Press `A` to enter the init pose.
- Press `A` + `B` to enter policy mode.
- Press `X` to unpause the motion flow.

Start with the publisher paused, press `A` to initialize and `A+B` to enter policy mode, then press `X` while standing naturally to begin live following.

At startup, the SMPL stream publishes an upright, arms-down reference constructed using the official skeleton FK. After live tracking starts, pressing `X` again holds the last SMPL body pose, wrist targets, and heading while simulation continues. The parallel GMR viewer / G1 stream returns to its default standing reference, so its paused pose can differ from the SMPL policy target.

## Optional: G1 robot-reference mode

To track the existing GMR-retargeted robot motion, use `checkpoints/sonic/release/g1/policy.yaml` and change the tracking flags to `--motion-backend zmq --motion-zmq-connect tcp://127.0.0.1:28701`. This mode does not require `--publish-smpl`; pressing `X` to pause returns to the default standing reference.

The furthest reference frame is 180 ms for SMPL and 900 ms for G1. The current configs add a 40 ms interpolation tolerance; tracking and execution contribute additional end-to-end delay. See [SONIC SMPL Input](/reference/sonic-smpl-input) for the input contract.

## DH116S hand control

The Pico publisher also sends the left and right analog trigger values as
normalized hand-grip commands on TCP port `5593`. The left trigger controls the
left DH116S and the right trigger controls the right DH116S.

Install the repo-local SDK once on the computer with the DH116S CANFD adapters:

```bash
./third_party/dh116s_sdk/install.sh
uv sync --project venv/dh116s
```

The installer detects `aarch64` or `x86_64` and writes the runtime under
`third_party/dh116s_sdk/python`. Then start the hand process from the root
project environment:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/dh116s --no-sync scripts/dh116s_control.py \
  --hand-dir double \
  --connect tcp://<pico-publisher-ip>:5593
```

:::warning
The hardware process enables and homes every requested hand during startup.
Clear the workspace around both hands before launching it. If either hand fails
to initialize in `double` mode, the process disconnects and exits.
:::

The default hardware mapping is left hand `can0` / node `1` and right hand
`can1` / node `1`. To validate the ZMQ stream and the conservative 40% grasp
mapping without importing the SDK or moving hardware, run:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_control.py --dry-run --hand-dir double
```

When the grip stream becomes stale, the process keeps the last commanded hand
pose. Stopping the process disconnects the SDK without automatically opening
the hands. Use `--hand-dir left` or `--hand-dir right` for staged bring-up.

## Notes

- `pico_retarget_pub.py` publishes the live motion stream consumed by the tracking policy and opens the retarget mjviser server.
- Hand-grip messages use their own ZMQ port and do not change the existing Pico button protocol.
- `sim2real/sim_env/base_sim.py` is the sim2sim execution backend.
- For real hardware, [Robot I/O](/reference/robot-io) lists the inline and bridge deployment modes.
- If the publisher and policy run on different machines, set both `--motion-zmq-connect tcp://<publisher_ip>:28702` and `--pico-zmq-connect tcp://<publisher_ip>:5592`.

## Next Steps

- [Motion Recording](/tutorials/motion-recording)
