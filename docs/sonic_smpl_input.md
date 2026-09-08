---
title: SONIC SMPL Input
slug: /reference/sonic-smpl-input
---

# SONIC SMPL Input

The default SONIC release from `GR00T-WholeBodyControl` uses SMPL mode for PICO full-body teleoperation. The included adapter is `checkpoints/sonic/release/smpl/policy.yaml`; its complete ONNX contains the encoder, finite scalar quantizer, and action decoder.

```text
raw XRobot global body rotations
    -> parent-relative SMPL local rotations
    -> official human_joints_info FK -> SMPL joint references
    -> elbow swing + wrist rotation mapping -> G1 wrist references
    -> SMPL ZMQ stream -> SONIC encoder and decoder
```

The publisher also generates a GMR robot reference for its viewer and the optional SONIC G1 mode. The SMPL policy's wrist targets come directly from the source SONIC mapping, rather than from GMR IK.

## Model inputs

The single-file ONNX accepts two flat float32 inputs and exposes `action[29]` and `token[64]`:

| Input | Size | Layout |
| --- | ---: | --- |
| `smpl_input` | 840 | All 10 frames of joints (720), then all orientations (60), then all wrist references (60). |
| `proprioception` | 930 | 10-frame histories of base angular velocity (30), joint position minus default (290), joint velocity (290), previous action (290), and projected gravity (30). |

The encoder reference consists of these groups, concatenated in this order:

| Observation class in `sonic.py` | Shape before flattening | Source |
| --- | ---: | --- |
| `sonic_smpl_joints_multi_future_local` | `[10, 72]` | `motion_data.smpl_joint_pos_root`, 24 canonical joint xyz positions. |
| `sonic_smpl_root_ori_b_multi_future` | `[10, 6]` | SMPL root relative to the current robot root after initial heading alignment. |
| `sonic_joint_pos_multi_future_wrist_for_smpl` | `[10, 6]` | Six named wrist fields in `motion_data.joint_pos`, replaced by the source SONIC wrist mapping. |

Reference frames are `[0, 1, ..., 9]` at 20 ms spacing. Rotation features flatten the first two rotation-matrix columns **row by row**: `[R00, R01, R10, R11, R20, R21]`. This differs from stacking complete columns. History initialization and joint ordering follow the source C++ deployer; the YAML selects `history_initialization: source_cpp` and `joint_order: policy` where required.

The export wrapper inserts these groups into the original universal encoder's 1762 fields, fixes mode to `2`, and zero-fills inactive fields. Runtime observations therefore do not need to construct the unused encoder modes. See the [artifact notes](https://github.com/mwondering/mimiclite-deploy/blob/main/checkpoints/sonic/release/README.md) for source hashes and validation.

## Stream and startup

Start `pico_retarget_pub.py` with `--publish-smpl`. The default skeleton is included at `checkpoints/sonic/release/smpl/human_joints_info.pkl`; use `--smpl-human-joints-info-path` only to select a different compatible skeleton. The tracking flags are `--motion-backend smpl_zmq --motion-zmq-connect tcp://127.0.0.1:28702`. Full launch commands are in [Pico Teleoperation](/tutorials/pico-teleoperation).

| Payload field | Shape | Meaning |
| --- | ---: | --- |
| `smpl_body_pose_aa` | `[N, 21, 3]` | Parent-relative local axis-angle body rotations, excluding the root. |
| `smpl_joint_pos_root` | `[N, 24, 3]` | Canonical joints rotated into the source SMPL root convention. |
| `smpl_root_quat_w` | `[N, 4]` | SMPL reference root quaternion in `wxyz` order after the source root-frame conversion. |
| `joint_pos` | `[N, 29]` | G1 joint fields; the six wrist values use the source SONIC mapping. Other joints are not consumed by the SMPL encoder. |

Before the first live frame, the publisher constructs a neutral upright, arms-down pose using the same official FK. It does not send zero joint positions as a substitute skeleton. After live tracking, `X` pause holds the last SMPL body pose, wrist reference, and heading; simulation continues. The parallel GMR stream / viewer returns to the default robot stand pose.

## Canonical skeleton computation

XRobot provides body poses as `[x, y, z, qx, qy, qz, qw]`. The body quaternions are global orientations, not local parent-relative rotations. The SMPL path converts them to local rotations before FK:

```text
raw global quaternions
    -> parent-relative local SMPL rotations
    -> smpl_body_pose_aa
    -> human_joints_info.pkl rest joints and parent tree
    -> canonical FK
    -> select joints [0..21, 39, 54]
    -> source root-axis conversion and inverse root rotation
    -> smpl_joint_pos_root
```

This preserves the source FK's root rest-position offset. Do not additionally subtract joint 0 or force the pelvis to zero: that changes the model's training input. Raw tracker body positions and GMR's `scaled_human_data` are IK target positions, not this canonical skeleton.

## Wrist mapping

The wrist reference order is:

```text
left_wrist_roll, right_wrist_roll,
left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

`sonic_wrist_targets_from_smpl_pose` follows the source pose-mode mapping: decompose each elbow into Y-axis twist and swing, extract intrinsic `XYZ` Euler angles, and combine elbow swing with the local wrist rotation. Left/right signs follow the source implementation. Identity elbows receive a finite zero-angle treatment.

`apply_sonic_wrist_targets` writes the six results into the SMPL payload by joint name. It leaves the parallel GMR reference unchanged. Human wrist/hand **positions** in the 24-joint skeleton and these six robot wrist **angles** carry different information; both are required by the released encoder.

## Relevant files

- `sim2real/teleop/smpl_stream.py`
- `sim2real/teleop/pico_retarget_pub.py`
- `sim2real/rl_policy/utils/motion_buffer.py`
- `sim2real/rl_policy/observations/sonic.py`
- `scripts/export_sonic_release.py`
- `checkpoints/sonic/release/smpl/policy.yaml`
