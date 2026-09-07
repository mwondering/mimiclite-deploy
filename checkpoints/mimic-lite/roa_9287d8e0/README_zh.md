# 官方 MimicLite-ROA / 9287d8e0

于 2026-09-07 从
[MimicLite 官方发布页](https://github.com/Roboparty/MimicLite/blob/3963976de8778d9292305fc8efacbcae79ed6685/README.md)
列出的部署目录下载。该模型为 16 × 16384 G1 mixture Huge PPO-ROA 策略，
经过 `train -> adapt -> finetune`，actor 隐藏层为 `[1024, 1024, 1024]`。

- [官方部署文件](https://drive.google.com/drive/folders/1AFcvP4oDbEskx-wip5bJN-JBaUvwp8MH)
- [官方训练 run](https://wandb.ai/elijahgalahad/mimic_lite/runs/9287d8e0)
- ONNX 的 Drive 文件 ID：`1xtHn7tu2s-A8RQ84AuT1EqiXQBYRe9pt`。
- YAML 的 Drive 文件 ID：`1z5qIRs78-k6syKcTEoiEvi-Jp3VBhaYy`。

两个文件均通过 `rclone copyurl` 下载，未作修改。以下 SHA-256 在本地计算，
用于记录此次下载内容，并非上游独立公布的校验值：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `policy.onnx` | 26,853,863 | `78aec8b2738f3940fb47a58021b9584faa3f69b53476551c52a2026909db33c1` |
| `policy.yaml` | 6,756 | `47be255e107d0447832b4819b555fe6c97208687b952f58b84338e6aeabf5069` |

## 启动与兼容性

使用 root 项目及 `inference-cpu`，先准备本地动作和 G1 资源缓存。
在仓库根目录运行：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/sim_env/integrated_sim2sim.py \
  --policy-config checkpoints/mimic-lite/roa_9287d8e0/policy.yaml \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --env-dt 0.02 --initial-pause-s 2
```

现有运行时提供 `command[304]` 和 `policy[535]` 两组输入；ONNX 输出
`action[29]` 和 `priv_student[1024]`。最远未来参考是 50 Hz 下的 4 帧，即 0.08 秒。
官方模型使用 IR 10 / opset 20。不支持该格式的旧版机载 ONNX Runtime 需要
另行验证格式转换或升级运行时；此次下载不包含兼容格式转换。

## 本地验证

已通过 ONNX checker、YAML 输入名匹配和 CPU 单步推理。
MuJoCo 冒烟测试使用 seed `20260728`、初始暂停 2 秒，以及
`datasets/root90/motions/backward__00__FairySteps.npz`（401 帧）。
运行了 15 秒仿真时间，包含末帧保持；轨迹保存在
`outputs/mimiclite_roa_9287d8e0/download_smoke_root.npz`。
最终相对 root XY 误差为 0.477 米。这段短测不能证明全面跟踪质量或真机安全性，
本次没有控制实体 G1。

此目录与两个 SP-Tracking checkpoint 一同包含在部署仓库中。
加载这些部署文件不需要源训练仓库或训练 `.pt`。
