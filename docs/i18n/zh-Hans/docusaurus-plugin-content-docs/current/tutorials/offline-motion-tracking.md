# Offline Motion Tracking

这个教程使用 root project 里的 tracking policy 和离线动作参考。

默认 motion：

```text
hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

## Sim2Sim

### 支持的本地 NPZ 格式

`--motion-path` 同时支持现有 any4hdmi 数据集（包括 HF 地址），以及包含
`fps`、`joint_pos`、`body_pos_w`、`body_quat_w` 的独立 IsaacLab/SP-Tracking
G1 NPZ 动作，无需改名或手动转换。未提供名称的 IsaacLab 导出默认采用
G1 的 29 关节 IsaacLab 顺序，并将 body 索引 0 视为 pelvis；如果文件包含
`joint_names` 和 `body_names`，则优先按名称映射。四元数必须采用 wxyz 顺序。

独立动作会转换到 `.cache/motion/isaaclab/`，原始文件不修改。加载器保留根部
轨迹和关节角，再用机器人 MJCF（或 `motion.mjcf_path` 覆盖值）重算身体位姿
和速度，并通过 any4hdmi 重采样到 50 Hz。原文件的非根部身体数组和速度数组
不会直接复制。单进程 integrated sim2sim 同样支持这一入口。

先启动 MuJoCo 执行进程。启动后终端会打印 mjviser URL：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

在第二个终端启动 tracking policy：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

两个进程各自负责：

- `sim2real/sim_env/base_sim.py` 在 MuJoCo 里执行 `low_cmd`，并发布 `low_state`
- `sim2real/rl_policy/tracking.py` 消费 `low_state`，跑导出的 policy，再发出下一帧 `low_cmd`

两个进程都起来后，在 policy 终端按 `]` 开始跟踪。虚拟 gantry / elastic band 的开关和长度在 mjviser UI 里调。

## Sim2Real

上真机前，先在 [Robot I/O](/reference/robot-io) 里选择部署路径。例如 tracking policy 仍然这样启动：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Integrated Sim2Sim

如果希望 policy 和 MuJoCo 在同一个进程里运行，用 integrated runner。它会立即加载 policy，把机器人设置到 motion 第一帧，等待 5 秒后开始跟踪；motion 结束后会停在最后一帧。这个 runner 默认关闭 elastic band，启动后也会打印 mjviser URL。

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

非可视化运行加 `--headless`。有浏览器 client 连接时，mjviser scene 会每个 env step 更新一次。在 mjviser 模式里，停在最后一帧后点击 `Restart motion` 按钮会回到第一帧，并重新执行等待、跟踪、停在最后一帧的流程。

如果要做定量评测，可以加 `--trajectory-output <path>.npz` 保存完整轨迹，再用
`scripts/tracking_experiment/` 里的脚本计算动作进度、全局根部跟踪和局部身体跟踪指标。

## Next Steps

- [Pico Teleoperation](./pico-teleoperation.md)
