---
title: SONIC SMPL Input
slug: /reference/sonic-smpl-input
---

# SONIC SMPL Input

`GR00T-WholeBodyControl` 的默认 SONIC 发布版本使用 SMPL 模式进行 PICO 全身遥操作。本仓库适配配置是 `checkpoints/sonic/release/smpl/policy.yaml`；完整 ONNX 包含 encoder、有限标量量化器（FSQ）和动作 decoder。

```text
XRobot 原始全局人体朝向
    -> 相对父节点的 SMPL 局部旋转
    -> 官方 human_joints_info FK -> SMPL 关节点参考
    -> 肘部 swing + 手腕旋转映射 -> G1 手腕参考
    -> SMPL ZMQ 流 -> SONIC encoder 和 decoder
```

Publisher 同时生成 GMR 机器人参考，供网页和可选的 SONIC G1 模式使用。SMPL policy 的手腕目标直接采用原 SONIC 映射，不再取 GMR IK 的手腕解。

## 模型输入

单文件 ONNX 接收两个一维 float32 输入，输出 `action[29]` 和 `token[64]`：

| 输入 | 维数 | 排列 |
| --- | ---: | --- |
| `smpl_input` | 840 | 先放全部 10 帧关节点（720），再放全部朝向（60），最后放全部手腕参考（60）。 |
| `proprioception` | 930 | 各 10 帧的机体角速度（30）、关节位置减默认值（290）、关节速度（290）、上一动作（290）、投影重力（30）。 |

Encoder 参考按以下顺序分组拼接：

| `sonic.py` 中的观测类 | 展平前形状 | 来源 |
| --- | ---: | --- |
| `sonic_smpl_joints_multi_future_local` | `[10, 72]` | `motion_data.smpl_joint_pos_root`，24 个标准骨架关节的 xyz。 |
| `sonic_smpl_root_ori_b_multi_future` | `[10, 6]` | 初始航向对齐后，SMPL 根朝向相对于当前机器人根朝向的旋转。 |
| `sonic_joint_pos_multi_future_wrist_for_smpl` | `[10, 6]` | `motion_data.joint_pos` 中按名称选出的六个手腕值，已替换为原 SONIC 映射。 |

参考帧为 `[0, 1, ..., 9]`，间隔 20 ms。旋转特征取矩阵前两列，**按行展开**为 `[R00, R01, R10, R11, R20, R21]`，不是逐列堆叠。历史初始化和关节顺序与原 C++ 部署器一致；YAML 在相应观测上设置 `history_initialization: source_cpp` 和 `joint_order: policy`。

导出包装将这些分组填回原始 universal encoder 的 1762 维输入，固定 mode 为 `2`，其余模式字段补零，运行时不需要构造无用模式输入。来源散列和验证结果见[部署文件说明](https://github.com/mwondering/mimiclite-deploy/blob/main/checkpoints/sonic/release/README_zh.md)。

## 数据流与启动

启动 `pico_retarget_pub.py` 时必须加 `--publish-smpl`。默认骨架文件已包含在 `checkpoints/sonic/release/smpl/human_joints_info.pkl`，只有需要使用其他兼容骨架时才指定 `--smpl-human-joints-info-path`。Tracking 使用 `--motion-backend smpl_zmq --motion-zmq-connect tcp://127.0.0.1:28702`。完整命令见 [Pico Teleoperation](/tutorials/pico-teleoperation)。

| Payload 字段 | 形状 | 含义 |
| --- | ---: | --- |
| `smpl_body_pose_aa` | `[N, 21, 3]` | 相对于父节点的局部 axis-angle 旋转，不含根节点。 |
| `smpl_joint_pos_root` | `[N, 24, 3]` | 按原 SMPL 根坐标约定旋转后的标准骨架关节点。 |
| `smpl_root_quat_w` | `[N, 4]` | 经原模型根坐标转换后的 SMPL 根四元数，采用 `wxyz` 顺序。 |
| `joint_pos` | `[N, 29]` | G1 关节字段，其中六个手腕值采用原 SONIC 映射；其余关节不供 SMPL encoder 使用。 |

首次实时帧到达前，publisher 用相同的官方 FK 构造直立、双臂下垂的中性参考，不用全零关节点代替骨架。实时运行后按 `X` 暂停，SMPL 流保持最后一个人体姿态、手腕参考和朝向，仿真继续运行。并行的 GMR 流 / 网页则回到默认机器人站姿。

## 标准骨架计算

XRobot 每个 body 的 pose 为 `[x, y, z, qx, qy, qz, qw]`。四元数描述全局朝向，不是相对父节点的局部旋转，因此 SMPL 链路先转换为局部旋转，再做 FK：

```text
原始全局四元数
    -> 相对父节点的 SMPL 局部旋转
    -> smpl_body_pose_aa
    -> human_joints_info.pkl 的静态关节与父节点树
    -> 标准骨架 FK
    -> 选择关节 [0..21, 39, 54]
    -> 原模型根轴向转换和根旋转的逆变换
    -> smpl_joint_pos_root
```

这里保留原 FK 的根关节静态位置偏移。不要额外减掉第 0 个关节，也不要强制将 pelvis 设为零，否则会改变模型训练时的输入。Tracker 原始 body positions 和 GMR 的 `scaled_human_data` 是 IK 目标位置，不是这套标准骨架。

## 手腕映射

手腕参考顺序为：

```text
left_wrist_roll, right_wrist_roll,
left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

`sonic_wrist_targets_from_smpl_pose` 对齐原项目 pose 模式的映射：将每侧肘部旋转分解为绕 Y 轴的 twist 和 swing，提取内禀 `XYZ` 欧拉角，再组合肘部 swing 与手腕局部旋转。左右符号与原实现一致；零角度肘部旋转采用有限值处理。

`apply_sonic_wrist_targets` 按关节名称将六个结果写入 SMPL payload，并保留并行 GMR 参考的原值。24 关节骨架中的人类手腕 / 手部**位置**，与这里六个机器人手腕**角度**表达不同信息，发布版 encoder 同时需要两者。

## 相关文件

- `sim2real/teleop/smpl_stream.py`
- `sim2real/teleop/pico_retarget_pub.py`
- `sim2real/rl_policy/utils/motion_buffer.py`
- `sim2real/rl_policy/observations/sonic.py`
- `scripts/export_sonic_release.py`
- `checkpoints/sonic/release/smpl/policy.yaml`
