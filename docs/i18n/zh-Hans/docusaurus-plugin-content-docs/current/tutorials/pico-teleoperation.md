# Pico Teleoperation

本教程将 PICO / XR 全身追踪接到 **SONIC release SMPL**，使用 root project 的 tracking policy 控制 G1。模型来自 `GR00T-WholeBodyControl` 默认 SONIC 发布版本，保留原 PICO 全身遥操作使用的 SMPL 模式。

本仓库提供完整模型和 `human_joints_info.pkl` 骨架文件；来源及验证结果见
[部署文件说明](https://github.com/mwondering/mimiclite-deploy/blob/main/checkpoints/sonic/release/README_zh.md)。
按键沿用本仓库的 `A`、`A+B`、`X`，启动命令进入全身 SMPL 跟踪，不启动原项目的 planner 或 VR 三点模式。

## 1. 启动 Pico retarget publisher

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --robot g1 \
  --actual-human-height 1.76 \
  --publish-hz 30 \
  --publish-smpl
```

将 `--actual-human-height` 改成实际身高（米）。必须开启 `--publish-smpl`，SMPL 动作流使用端口 `28702`。

Publisher 的 mjviser 网页显示并行生成的 GMR 机器人参考，可用于检查追踪连接；SONIC SMPL 实际消费人体骨架与手腕参考，二者不是同一个表示。确保 PICO 全身数据和手柄数据 streaming 都已开启。

## 2. 选择执行后端

### Sim2Sim

启动 MuJoCo 执行进程：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run sim2real/sim_env/base_sim.py --robot g1
```

在另一个终端，把 tracking policy 接到实时 motion stream：

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

上真机前，先在 [Robot I/O](/reference/robot-io) 里选择部署路径。Pico 相关的 policy 参数保持一样：

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

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Pico 按键

- 按 `A` 进入 init pose。
- 同时按 `A` + `B` 进入 policy mode。
- 按 `X` 解除 motion flow 暂停。

先让 publisher 保持暂停，按 `A` 初始化，再按 `A+B` 进入策略模式；自然站立后按 `X` 开始实时跟随。

首次启动时，SMPL 流发布由官方骨架 FK 构造的直立、双臂下垂参考。实时运行后再次按 `X`，SMPL 流保持最后一个人体姿态、手腕参考和朝向，仿真继续运行。并行的 GMR 网页 / G1 动作流会回到默认站姿，因此暂停后网页参考与 SMPL policy 的目标可能不同。

## 可选：G1 机器人参考模式

如果要让 SONIC 跟踪现有 GMR 重定向的机器人动作，改用 `checkpoints/sonic/release/g1/policy.yaml`，并把 tracking 参数改为 `--motion-backend zmq --motion-zmq-connect tcp://127.0.0.1:28701`。此模式不需要 `--publish-smpl`，`X` 暂停会返回默认站姿参考。

SMPL 模式的最远参考帧为 180 ms，G1 模式为 900 ms；当前配置还增加 40 ms 插值容差，实际端到端延迟另含追踪与执行时间。输入细节见 [SONIC SMPL Input](/reference/sonic-smpl-input)。

## DH116S 灵巧手控制

Pico publisher 还会把左右扳机的连续值归一化为 `[0,1]` 的手部 grip command，
并通过 TCP `5593` 端口发送。左扳机控制左侧 DH116S，右扳机控制右侧 DH116S。

在连接了 DH116S CANFD 适配器的电脑上，先把 SDK 安装到当前 repo：

```bash
./third_party/dh116s_sdk/install.sh
uv sync --project venv/dh116s
```

安装脚本会自动识别 `aarch64` 或 `x86_64`，并把运行文件放到
`third_party/dh116s_sdk/python`。然后使用根项目环境启动：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/dh116s --no-sync scripts/dh116s_control.py \
  --hand-dir double \
  --connect tcp://<pico-publisher-ip>:5593
```

:::warning
硬件进程启动时会 enable 并自动 home 所有选中的手。启动前必须清空双手周围空间。
在 `double` 模式下，只要任意一只手初始化失败，进程就会断开并退出。
:::

默认硬件映射为左手 `can0` / node `1`，右手 `can1` / node `1`。如果只想验证
ZMQ 和最大 40% 的安全闭合映射，而不 import SDK 或移动硬件，运行：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_control.py --dry-run --hand-dir double
```

grip stream 断流后，进程会保持最后一次下发的手部姿态。停止进程时只断开 SDK，
不会自动张开。分阶段 bring-up 时可使用 `--hand-dir left` 或 `--hand-dir right`。

## Notes

- `pico_retarget_pub.py` 发布实时 motion stream 给 tracking policy 使用，并自己创建 retarget mjviser server
- hand grip 使用独立的 ZMQ 端口，不会改变现有 Pico 按键协议
- `sim2real/sim_env/base_sim.py` 是 sim2sim 的执行后端
- 真机部署时，[Robot I/O](/reference/robot-io) 里列出了 inline 和 bridge 两类方式
- 如果 publisher 和 policy 跑在不同机器上，同时设置 `--motion-zmq-connect tcp://<publisher_ip>:28702` 和 `--pico-zmq-connect tcp://<publisher_ip>:5592`

## Next Steps

- [Motion Recording](./motion-recording.md)
