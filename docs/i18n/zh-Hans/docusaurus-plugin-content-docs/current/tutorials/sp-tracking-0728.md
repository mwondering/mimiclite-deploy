---
title: 部署 SP-Tracking 0728
slug: /tutorials/sp-tracking-0728
---

# 部署 SP-Tracking 0728

此配置通过配套的完整 ONNX 运行
`0728_baoshou_waist_dataclean_changedr/checkpoint_22000.pt`。
sim2sim 和 G1 实机部署共用
`checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml`。
专用启动入口在运行前校验源 checkpoint 身份及适配 ONNX 的 SHA-256。
原有本地目录 `checkpoints/sp-tracking/spv5_2/` 中的模型权重相同；
新目录通过完整名称明确区分这次训练。

仓库包含部署 ONNX、JSON 元数据、YAML 和运动学 XML，不包含训练 `.pt`
和动作数据集。运行已准备好的策略无需训练仓库。

## Sim2sim

使用安装了 CPU 推理 extra 的根项目 Python 环境。将 G1 动作放到
`datasets/lafan40/motions/walk1_subject1.npz`，也可通过 `--motion-path`
指定其他本地 any4hdmi G1 NPZ。默认动作可在联网 PC 上准备：

```bash
uv run --no-sync hf download elijahgalahad/any4hdmi-g1-lafan \
  motions/walk1_subject1.npz --local-dir datasets/lafan40
```

在仓库根目录运行：

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim
```

此命令打开集成 MuJoCo 窗口，将机器人初始化到动作第 0 帧；
初始暂停两秒期间策略持续运行，随后开始播放动作。播放结束后策略持续跟踪末帧。
物理仿真步长为 5 ms，策略与参考动作步长为 20 ms（50 Hz）。
如果环境尚未设置，启动入口会设置 `SIM2REAL_ORT_NUM_THREADS=1`。

无界面运行并保存根节点轨迹：

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --headless --run-once \
  --root-trajectory-output outputs/sp_tracking_0728/root.npz
```

`--run-once` 表示到达末帧后退出。验证末帧持续保持时应省略此参数，
可用 `--max-runtime-s` 限定仿真时长。完整参数见 `sim2sim --help`。

## G1 sim2real

G1 使用根项目环境，按照 [Robot I/O](../robot_io.md) 安装
`inference-cpu` 和 `robot-g1`。将仓库、部署模型、选定动作及所需的离线机器人
资产缓存复制到 G1。在机器人用户的 `~/.bashrc` 中持久保存：

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

先检查参数；此命令不会建立 DDS 连接或发送机器人命令：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 --dry-run
```

在机器人上播放本地 NPZ：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --record --record-output outputs/sp_tracking_0728/real.npz
```

将 `eth0` 替换为 G1 电机网络所在网卡。此命令使用 inline G1 I/O 和宇树手柄：
`A` 进入默认姿态，`R1` 启用策略，`B` 播放／暂停参考动作，`R2` 进入零力矩模式。
若使用终端键盘，指定 `--controller keyboard`，对应按键为 `i`、`]`、空格、`o`。

接入已有 G1 ZMQ 动作发布端（包括 Pico 重定向）：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2real \
  --robot-interface eth0 --motion-backend zmq --controller pico
```

发布端按 [Pico 遥操作教程](pico-teleoperation.md) 启动。
专用入口在导入 runtime 前就设置两个 Hugging Face 离线变量。
默认使用 CPU 推理；目标机器已安装并验证相应后端时，可指定
`--inference-backend onnx-gpu` 或 `tensorrt`。
`--dry-run` 不验证 DDS 连通性或机载推理速度。本次改动未执行实机动作测试。

## 观测和动作约定

ONNX 保留完整 actor、参考编码器、估计器和学习得到的归一化模块。
通过图内 Concat 将原始扁平输入替换为以下四个语义输入，顺序与源实现一致：

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `robot_root_quat` | `1 × 4` | 骨盆四元数，wxyz 顺序。 |
| `estimator_history` | `1 × 6100` | 50 步历史，按观测项分组且旧帧在前：相对默认姿态的关节角、关节速度、重力、机身角速度、上一动作、最新实测关节力矩。 |
| `reference_encoder_input` | `1 × 1900` | 偏移 −42 至 +7 的 50 帧参考：世界系根位置、按列展开的 6D 旋转、关节角。 |
| `robot_key_body` | `1 × 195` | 根据实测 q/dq/gyro 和匹配训练的正运动学计算 13 个语义关键点。 |

四组输入共享历史状态，每个策略步只更新一次；部署关闭训练观测噪声。
输出为源 G1 MuJoCo 顺序的 29 维关节动作，裁剪到 ±10，再按 YAML 中与源实现一致的
动作缩放、默认姿态和 PD 增益执行。未来参考需求为 0.14 秒。
实机输入来自 IMU、关节编码器和最新电机力矩；此 actor 不需要实测全局根位置／速度
或外部动捕。

模型使用 ONNX IR 8 / opset 18。源导出 SHA-256 为
`65283dfa5f48c51dc28d2873852ecf379b04c2d0a97a23a4bb5b924a486684f2`；
适配模型 SHA-256 为
`c88d39cfe823c801dd7ada9d029360744dc692af900ce955824d6904cb978ef4`。

## 重建部署文件

```bash
uv run --no-sync python scripts/adapt_sp_tracking_spv5_2.py \
  --checkpoint-dir ../motion_tracking_sim2real_self/ckpts/0728_baoshou_waist_dataclean_changedr \
  --iteration 22000 \
  --output-dir checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr \
  --equivalence-trials 20
```

20 次 CPU 源模型／适配模型比较均通过，动作误差严格为 0。
验证命令和仿真指标见[实验记录](sp-tracking-0728-validation.md)。
