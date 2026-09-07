---
title: Run External Policies
slug: /tutorials/run-external-policies
---

# Run External Policies

The modular design of sim2real lets the same runtime execute different tracking
policies as long as they expose a compatible deploy YAML and ONNX model. We have
already converted several external policies into this format, so they can often
be interchanged by keeping the normal deploy command and only replacing
`--policy-config` with the policy YAML.

## Converted Checkpoints

Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder first, then use any checkpoint path below as the `--policy-config`
value.

| Policy | Checkpoint YAML | Notes |
| --- | --- | --- |
| Mimic-Lite v1.1 | `checkpoints/mimic-lite/v1_1/policy.yaml` | Full-scale T16 PPO-ROA finetune student. |
| HEFT PMG | `checkpoints/heft/pmg/policy.yaml` | Normal G1 motion stream. |
| HEFT Compliance | `checkpoints/heft/compliance/policy.yaml` | Normal G1 motion stream; compliance flag is forced off in the observation. |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | Normal G1 motion stream. |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Normal G1 motion stream. |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Requires the checkpoint-specific MJCF override for ZMQ publishers. |
| ScaleBFM M | `checkpoints/scalebfm/humanoid_transformer_m/policy.yaml` | Normal G1 motion stream. |
| ScaleBFM XL | `checkpoints/scalebfm/humanoid_transformer_xl/policy.yaml` | Normal G1 motion stream. |
| SONIC release G1 | `checkpoints/sonic/release/g1/policy.yaml` | Normal G1 motion stream. |
| SONIC release SMPL | `checkpoints/sonic/release/smpl/policy.yaml` | Uses `motion_backend: smpl_zmq` and the SMPL publisher. |
| SONIC v1.1 G1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | G1 motion stream with heading-normalized reference orientation. |
| SONIC low-latency G1 | `checkpoints/sonic/low_latency/g1/policy.yaml` | Normal G1 motion stream with the low-latency checkpoint. |
| SONIC low-latency SMPL | `checkpoints/sonic/low_latency/smpl/policy.yaml` | Four-frame SMPL input horizon. |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | Requires the official 1.64 GB ONNX artifact. |
| TWIST2 | `checkpoints/twist2/policy.yaml` | Normal G1 motion stream. |
| SP-Tracking SPV5-2 | `checkpoints/sp-tracking/spv5_2/policy.yaml` | Generated locally from an SPV5-2 export; normal G1 motion stream. |
| SP-Tracking SPV5-2A / 105000 | `checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml` | Included deploy ONNX; use the ordinary runtime with this YAML, not the hash-pinned 0728 launcher. |

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot-io inline \
  --motion-backend zmq \
  --controller pico \
  --policy-config checkpoints/heft/pmg/policy.yaml
```

This applies to ordinary G1 tracking policies that consume the normal G1 motion
stream, such as HEFT, TeleopIT, Humanoid-GPT, ScaleBFM, HoloMotion, and
standard any4hdmi / SONIC G1 motion policies.

## Policy-Specific Runtime Requirements

Some adapted policies need a different motion source or extra runtime asset.

### SP-Tracking SPV5-2

For the included **0728 / 22000** checkpoint, use the dedicated
[sim2sim and G1 deployment guide](sp-tracking-0728.md).

Adapt a source export directory containing matching
`policy_<iteration>.onnx`, `policy_<iteration>.json`, and
`checkpoint_<iteration>.pt` files:

```bash
uv run scripts/adapt_sp_tracking_spv5_2.py \
  --checkpoint-dir /path/to/sp_tracking/ckpts/run_name \
  --iteration 22000
```

The adapter writes `policy.onnx`, `policy.json`, `policy.yaml`, and a README to
`checkpoints/sp-tracking/spv5_2/`. It keeps every learned node and weight unchanged
and replaces only the original flat 8199-D input with four semantic inputs. CPU
ONNX Runtime equivalence checks are enabled by default. The runtime observation
implementation reproduces the 50-step estimator history, 50-frame reference
window (`-42..7`), and 13-key-body feature contract without changing existing
observation groups. The furthest future reference is seven 50 Hz frames, so the
policy's motion-lookahead latency is 0.14 s.

The source PyTorch checkpoint is checked for a matching iteration but is not
copied into the runtime artifact. This keeps training state separate from the
deploy checkpoint.

### HoloMotion v1.4.0

Download the official ONNX without modifying it:

```bash
mkdir -p checkpoints/holomotion/v1_4_0
wget -O checkpoints/holomotion/v1_4_0/policy.onnx \
  https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx
```

The expected SHA-256 is
`859174937272747e762075db482e2b8d05d40dacb3a09884fc9d7d42086bbffe`.

### BFM-Zero

BFM-Zero needs its checkpoint-specific MJCF for the MuJoCo FK used by its motion
observations. For direct NPZ playback this is stored in the policy YAML. For ZMQ
publishers, pass the same MJCF override to the publisher.

BFM-Zero is compute-heavy. Prefer `--inference_backend onnx-gpu` for policy
inference when CUDA ONNX Runtime is available. Use `onnx-cpu` only as a
compatibility fallback on hosts without a working GPU provider.

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy_config checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml \
  --inference_backend onnx-gpu
```

```bash
uv run sim2real/teleop/npz_pub.py \
  --motion_path ../any4hdmi/output/g1/lafan/motions/walk1_subject1.npz \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml
```

### SONIC SMPL Mode

SONIC SMPL mode is not the normal G1 `motion_backend=zmq` stream. Use the SONIC
SMPL policy config, keep its `motion_backend: smpl_zmq` setting, or pass
`--motion-backend smpl_zmq`, and run the SMPL/XRobot publisher path.

Minimal sim2sim Pico test:

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py --publish-smpl
```

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sonic/release/smpl/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --controller pico
```

See [SONIC SMPL Input](/reference/sonic-smpl-input) for the data contract.

## Hardware Notes

Current notes from the G1 test setup:

- BFM-Zero works with the MJCF override.
- TeleopIT can walk well, but joint chatter has been observed and double-knee
  kneeling is not reliable yet; treat this as a deploy-infra / policy
  compatibility item before using that behavior on hardware.
- HEFT has shown light chatter but strong overall tracking behavior.
