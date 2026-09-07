# SP-Tracking SPV5-2A / 105000

源训练：`2026-09-06_02-31-13_SPTracking-G1-BFM-SPV5-2AActor-HEFTCritic-HEFTReward_4gpu_8192env_motion_data_correct`
源训练步数：`105000`

`policy.onnx` 保留完整源 actor，并增加四个语义输入。
训练 `.pt` 仅用于来源核验，不会复制到部署目录。

部署 ONNX SHA-256：
`9a4f37c1e6226a8749da2ce931bd8dfc796ff29746b67bea3f4bcd8a912e8aaf`。
大小：64,830,945 字节（61.83 MiB）。请使用下面的通用入口；
`scripts/run_sp_tracking_0728.py` 有意限制为只能加载 0728 模型。

## Sim2sim

在仓库根目录运行，先安装 `inference-cpu`，准备本地 G1 any4hdmi 动作及
离线机器人资源缓存。资源准备方法见
[0728 配置说明](../../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/sp-tracking-0728.md)。

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/sim_env/integrated_sim2sim.py \
  --policy-config checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --env-dt 0.02 --initial-pause-s 2
```

加入 `--headless --run-once` 可无界面运行，并在动作末帧退出。
不加 `--run-once` 时，仿真继续跟踪并保持最后一帧。

## G1 sim2real

安装 root 项目的 `inference-cpu` 和 `robot-g1` extras，准备本地动作和资源缓存，
参见 [Robot I/O](../../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/robot_io.md)。
将 `HF_HUB_OFFLINE=1` 和 `HF_HUB_DISABLE_TELEMETRY=1` 的 export 持久化到
机器人用户的 `~/.bashrc`。将 `eth0` 换成已确认的电机通信网口。
以下命令会连接真机 I/O；应完成 G1 常规安全检查，准备好支撑和急停人员后再启动。

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/rl_policy/tracking.py \
  --robot g1 --robot-io inline --robot-interface eth0 --controller joystick \
  --policy-config checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml \
  --motion-backend npz --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --rl-rate 50
```

手柄：`A` 默认姿态，`R1` 策略模式，`B` 暂停/继续参考动作，`R2` 零力矩。
零力矩不能代替物理支撑。

## 验证（2026-09-07）

已通过 ONNX checker、元数据/哈希核验和 CPU 单步推理。
无界面 integrated sim2sim 完成了
`datasets/root90/motions/backward__00__FairySteps.npz` 的全部 401 帧，
初始暂停 2 秒，seed 为 `20260728`。本地轨迹保存在
`outputs/sp_tracking_105000/upload_smoke_root.npz`（不提交）。
现有导出元数据记录了 20 次源模型与适配模型的零误差比较；本次上传没有重跑源模型比较。
这只是冒烟测试，不代表全面跟踪质量评估或真机安全验证。本次未执行真机控制。
