# Sim2Sim 运行原理与操作指南：离线 Play 和 PICO 遥操作

本文基于当前仓库源码，说明 MuJoCo sim2sim 的运行入口、策略控制循环、离线动作播放，以及 PICO 遥操作的数据流。

本文中的 **play** 指离线动作播放与策略跟踪，相关入口是 `tracking.py` 和 `integrated_sim2sim.py`。以下命令均在仓库根目录运行。

> 验证范围：本文来自源码检查，没有实际安装依赖、启动仿真或连接 PICO 验证。当前目录包含 SP-Tracking 0728 的 YAML 和 ONNX；文档旧示例中的 `checkpoints/mimic-lite/v1_1/` 和本地 `datasets/` 目录在检查时不存在，因此本文使用现有 SP-Tracking 模型举例。

## 1. 整体原理

Sim2sim 的核心是：**用已经训练好的策略，根据“机器人当前状态 + 参考动作”，计算关节目标，再让 MuJoCo 通过 PD 控制执行。**

离线 play 和 PICO 遥操作共用控制链，区别主要在参考动作来源：

- **离线 play**：参考动作来自动作文件。
- **PICO 遥操作**：参考动作来自人体追踪，经过实时重定向后送给策略。

```mermaid
flowchart LR
    A["离线动作文件"] --> C["Tracking：参考动作处理"]
    B["PICO 人体追踪"] --> R["重定向 Publisher"]
    R -->|ZMQ 动作流| C
    C --> O["构造策略观测"]
    S["MuJoCo 机器人状态"] -->|low_state| O
    O --> P["ONNX 策略推理"]
    P --> Q["关节目标 q_target"]
    Q -->|low_cmd| D["PD 控制与力矩限幅"]
    D --> M["MuJoCo 物理仿真"]
    M --> S
```

需要区分三种“动作”：

| 名称 | 含义 |
| --- | --- |
| 参考动作 `motion` | 希望机器人模仿的关节姿态、身体位置和朝向等 |
| 策略输出 `action` | 网络根据参考和当前状态算出的控制量 |
| 关节目标 `q_target` | 把 `action` 缩放、加上默认关节角后得到的 PD 目标 |

PICO 重定向出来的关节角不会直接作为执行机器人的每一步关节目标。中间还有 tracking policy，通过反馈控制实现动作跟踪和平衡。

## 2. 常规 sim2sim：两个独立进程

主要入口：

- [base_sim.py](sim2real/sim_env/base_sim.py)：MuJoCo 物理执行和浏览器可视化。
- [tracking.py](sim2real/rl_policy/tracking.py)：参考动作读取、观测构造、策略推理和控制命令发送。

默认通信如下：

| 数据 | 方向 | 默认端口 |
| --- | --- | --- |
| `low_state` | 仿真 → policy | TCP `5590` |
| `low_cmd` | policy → 仿真 | TCP `5591` |
| 实时参考动作 | PICO publisher → policy | TCP `28701` |
| PICO 按钮状态 | PICO publisher → policy | TCP `5592` |
| 手部 grip | PICO publisher → 独立手部控制进程 | TCP `5593` |

这些数据使用 ZMQ 传递。机器人状态通道和参考动作通道是两套独立连接。

因此，这两个参数控制不同事情：

```bash
--robot-io zmq
--motion-backend zmq
```

`--robot-io zmq` 表示通过 ZMQ 收发机器人的状态和控制命令；`--motion-backend zmq` 表示通过 ZMQ 接收参考动作。

离线播放通常使用：

```bash
--robot-io zmq --motion-backend npz
```

PICO 遥操作使用：

```bash
--robot-io zmq --motion-backend zmq
```

## 3. 策略每个周期具体做什么

核心执行代码在 [BasePolicy.step()](sim2real/rl_policy/base_policy.py)，默认频率为 **50 Hz**，即每 20 ms 执行一次。

一次循环可以概括为：

```text
处理键盘或 PICO 的模式切换
→ 读取最新机器人状态
→ 更新参考动作和观测历史
→ 构造网络输入
→ 运行 ONNX
→ 将 action 转换为关节目标
→ 根据 init / zero / policy 模式选择实际目标
→ 发送 low_cmd
```

### 3.1 机器人状态

机器人反馈主要包含：

- 根部四元数。
- 根部角速度。
- 关节角。
- 关节速度。
- 关节力矩。

普通仿真的 `low_state` 消息并不直接携带完整的全局根部位置，不能理解为向策略暴露全部 MuJoCo 真值。

[StateProcessor](sim2real/rl_policy/utils/state_processor.py) 将后端读取到的状态复制到观测使用的数组中。

### 3.2 观测由 YAML 决定

观测不是在主循环里写死的，而是由 `policy.yaml` 中的 `observation` 配置决定。运行时根据 `_target_` 找到对应观测类，再按配置顺序拼接输入。

当前模型配置是：

[SP-Tracking policy.yaml](checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml)

它配置了四组输入：

| 输入组 | 用途 |
| --- | --- |
| `robot_root_quat` | 机器人根部四元数 |
| `estimator_history` | 状态估计所需的历史输入 |
| `reference_encoder_input` | 参考动作窗口 |
| `robot_key_body` | 机器人关键身体部位特征 |

对应实现位于 [sp_tracking.py](sim2real/rl_policy/observations/sp_tracking.py)。其他模型可以配置不同的观测组合，因此更换模型时需要使用与该模型匹配的 YAML。

### 3.3 网络输出如何成为关节目标

对于受控关节，转换公式是：

```text
q_target = default_joint_pos + action_scale × action
```

其中：

- `default_joint_pos`：动作输出的默认姿态基准。
- `action_scale`：网络输出到关节角偏移的缩放。
- `policy_joint_names`：网络输出对应的受控关节顺序。
- `clip_actions`：可选的网络输出裁剪范围。

当前 SP-Tracking 配置的动作裁剪范围是 `[-10, 10]`。

**参考动作关节角是网络输入，默认关节角是动作输出的基准，两者不是同一个东西。**

实现还会接收网络返回的 `next` 状态，并将其写回运行状态字典，为带内部状态的导出模型提供支持。

## 4. MuJoCo 如何执行关节目标

[SimulationBridge](sim2real/sim_env/utils/bridge.py) 收到的命令包含：

```text
q_target、dq_target、tau_ff、kp、kd
```

仿真每个物理步计算：

```text
tau = tau_ff
    + kp × (q_target - q_current)
    + kd × (dq_target - dq_current)
```

随后执行关节力矩限幅，将结果写入 `mj_data.ctrl`，再调用：

```python
mujoco.mj_step(mj_model, mj_data)
```

当前 tracking 主循环发送的目标速度和前馈力矩都是零，因此通常可以理解为：

```text
tau = kp × 位置误差 - kd × 当前关节速度
```

PD 增益来自模型配置，由 [ActionManager](sim2real/rl_policy/utils/command_sender.py) 放入控制命令。

### 4.1 默认频率

| 环节 | 默认频率 |
| --- | --- |
| MuJoCo 物理步进、PD 计算 | 200 Hz，`sim_dt=0.005` |
| Tracking 策略循环 | 50 Hz，`rl_rate=50` |
| `base_sim` viewer 同步调用 | 每 4 个物理步一次，约 50 Hz |
| PICO publisher | 30 Hz |

常规双进程中，policy 和仿真分别按自己的时钟调度。平均每条策略命令覆盖约 4 个物理步，但没有严格的“一次推理后必须执行四步”的同步屏障。

`base_sim.py` 的 `decimation=4` 在这里控制 **viewer 同步间隔**；policy 频率由另一个进程的 `--rl-rate` 决定。

## 5. 离线 play 的启动和操作

### 5.1 准备根项目环境

推理、policy 和仿真使用根项目环境：

```bash
uv sync --extra inference-cpu
```

以下示例使用现有 SP-Tracking 模型和配置中采用的 HF 动作地址。

### 5.2 终端 1：启动仿真

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

打开终端打印的 mjviser URL，可以看到 MuJoCo 物理仿真。

### 5.3 终端 2：启动离线 tracking

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml \
  --robot-io zmq \
  --motion-backend npz \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz \
  --controller keyboard \
  --rl-rate 50
```

`hf://...` 由资产加载组件解析；运行时仍需要相关动作和机器人模型可获取或已经缓存。也可以换成本地兼容动作数据路径。

### 5.4 在 policy 终端操作

| 按键 | 源码中的实际行为 |
| --- | --- |
| `i` | 进入 `init`，逐步靠近模型配置的默认关节姿态 |
| `]` | 重置参考动作和观测，进入 `policy`；参考动作处于暂停状态 |
| 空格 | 切换离线参考动作的暂停/继续 |
| `o` | 进入 `zero`，使用当前关节角作为目标 |
| `Ctrl+C` | 退出当前进程 |

通常操作顺序：

```text
启动仿真和 policy
→ i 初始化
→ ] 进入策略控制
→ 空格开始播放参考动作
```

**旧文档只写了按 `]` 开始跟踪，但当前源码里还需要按空格，参考时间才会推进。** `]` 后策略已经在控制机器人，只是参考停在第一帧。

`zero` 不能按字面理解为“所有电机零力矩”：它仍使用配置中的 PD 增益，只是目标位置设为当前反馈位置。

### 5.5 离线参考如何推进

参考处理代码位于 [tracking.py](sim2real/rl_policy/tracking.py)，加载适配代码位于 [motion.py](sim2real/rl_policy/utils/motion.py)。

处理流程：

1. 用 any4hdmi 加载动作，默认转到 50 FPS。
2. 选择一段 motion。
3. 使用 `motion_t` 作为当前帧索引。
4. 按 `future_steps` 取历史、当前和未来参考帧。
5. 未暂停时，每次策略循环推进一帧。
6. 到末尾后停在最后一帧，不自动循环。

暂停时，普通 `Tracking` 会把参考窗口收拢到当前帧，并将参考速度清零。**暂停的是参考播放，物理仿真和策略控制仍然继续。**

当前 SP-Tracking 使用 50 Hz 参考契约，不应随意通过修改 `--rl-rate` 改变播放速度，否则参考时间、历史窗口和训练时序可能不再匹配。

## 6. Viewer 和虚拟吊带

`base_sim.py` 默认开启虚拟弹性吊带，向机器人施加一个连接上方固定点的弹簧阻尼力。实现位于 [elastic_band.py](sim2real/sim_env/utils/elastic_band.py)。

界面提供：

- `Elastic Band / Enabled`：启停吊带。
- `Length`：调节吊带长度。
- `Reset sim`：重置物理仿真。

吊带能帮助初始化和观察动作，但会改变机器人受到的外力。判断策略是否能自主站稳、行走时，需要观察关闭吊带后的表现。

**`Reset sim` 只重置仿真，不会同步重置另一个进程里的参考动作进度。** 跌倒后重新开始，通常还需要在 policy 侧重新初始化，再按 `]` 和空格。

## 7. 单进程 integrated play

入口是 [integrated_sim2sim.py](sim2real/sim_env/integrated_sim2sim.py)。

它把策略和 MuJoCo 放在同一进程，通过内存传递状态和命令，适合离线观察及定量评测。

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy-config checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

运行流程：

```text
加载模型和动作
→ 把机器人初始化到参考第一帧
→ 在第一帧参考下运行策略，默认等待 5 秒
→ 自动推进动作
→ 每推理一次，执行 4 个物理步
→ 动作结束后停在最后一帧
```

### 7.1 与常规双进程的区别

| 项目 | 双进程 `base_sim + tracking` | 单进程 integrated |
| --- | --- | --- |
| 状态与命令传递 | ZMQ | 内存 |
| 时序 | 各自调度 | 固定推理/物理步比例 |
| 开始方式 | 按键操作 | 初始等待后自动开始 |
| 初始化 | 手动模式切换 | 设置到动作第一帧 |
| 虚拟吊带 | 默认开启 | 不使用 |
| PICO 实时流 | 支持 | 当前只支持离线 `npz` |

Integrated 默认 `env_dt=0.02`、`sim_dt=0.005`，因此每次策略推理后严格执行 4 个物理步。这里的 `decimation` 是物理步与策略步的比例，与 `base_sim.py` 中控制 viewer 间隔的同名参数需要区分。

### 7.2 常用参数

无界面播放一遍后退出：

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy-config checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz \
  --headless \
  --run-once \
  --trajectory-output /tmp/tracking_trajectory.npz
```

- `--headless`：不启动 viewer，不按墙钟实时限速；初始等待按仿真时间计算。
- `--run-once`：动作到末尾后退出。
- `--initial-pause-s`：修改初始等待时长。
- `--trajectory-output`：保存参考与执行轨迹。

可视化模式在最后一帧暂停后，可以点击 `Restart motion` 从头再来。

项目还提供 [run_sp_tracking_0728.py](scripts/run_sp_tracking_0728.py)，用于固定 0728 模型的启动和校验。该脚本默认引用本地 `datasets/lafan40/motions/walk1_subject1.npz`；当前目录没有该数据目录，使用时需要准备数据或通过 `--motion-path` 覆盖。

## 8. PICO 遥操作的数据链

PICO 需要在两个主进程外增加一个 publisher：

```text
PICO 全身追踪
→ XRoboToolkit PC Service / SDK
→ XRobotStreamer
→ GMR 人体到 G1 的重定向
→ 标准参考动作 ZMQ 流
→ Tracking
→ ONNX
→ MuJoCo PD 执行
```

入口是 [pico_retarget_pub.py](sim2real/teleop/pico_retarget_pub.py)。

### 8.1 Publisher 的处理过程

1. 获取人体部位位姿和手柄输入。
2. 对齐人体参考的水平位置与朝向。
3. 处理地面高度偏移。
4. 用 GMR 将人体姿态重定向到 G1。
5. 通过机器人正向运动学得到关节及身体位姿。
6. 发布带时间戳、关节名称和身体名称的参考动作。
7. 在独立 viewer 中显示重定向结果。

默认人体身高为 `1.6 m`，通过 `--actual-human-height` 修改。

正常重定向路径发布的数据包括关节位置、身体世界位置、身体世界四元数，以及时间戳、暂停状态和动作段起始标记等信息。

### 8.2 两个 viewer 各自代表什么

| Viewer | 看到的内容 |
| --- | --- |
| PICO publisher 的 viewer | 重定向后的参考姿态 |
| `base_sim` 的 viewer | 经策略和物理仿真执行后的机器人 |

前一个能正确摆动作，只说明参考链路工作正常；后一个才体现策略的跟踪和平衡效果。

## 9. PICO 的启动和操作

### 9.1 环境准备

遥操作使用独立环境：

```bash
uv sync --project venv/pico
```

此外还需要 XRoboToolkit PC Service、Python SDK，以及 PICO 端全身追踪校准和 streaming。具体安装步骤见 [Teleop PC 环境文档](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/teleop-x86-64.md)。

环境分工：

| 进程 | Python 环境 |
| --- | --- |
| `base_sim.py` | 根项目 |
| `tracking.py` | 根项目 |
| `integrated_sim2sim.py` | 根项目 |
| `pico_retarget_pub.py` | `venv/pico` |

### 9.2 终端 1：启动 PICO publisher

将身高参数改为操作者实际身高：

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --actual-human-height 1.6
```

### 9.3 终端 2：启动物理仿真

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

### 9.4 终端 3：启动实时 tracking

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml \
  --robot-io zmq \
  --motion-backend zmq \
  --controller pico \
  --rl-rate 50
```

这是依据现有配置和通用实时接口给出的连接方式，本文未实际连接 PICO 验证该模型的遥操作表现。

### 9.5 PICO 按键

| 按键 | 作用位置 | 行为 |
| --- | --- | --- |
| `A` | policy | 进入初始化姿态模式 |
| `A+B` 同时按 | policy | 进入策略控制模式 |
| `B` 单独按 | policy | 进入 `zero` 模式 |
| `X` | publisher | 切换实时人体动作与暂停参考 |

通常操作顺序：

```text
A 初始化
→ A+B 进入策略控制
→ X 开始实时动作流
```

模式切换实现见 [control_mode.py](sim2real/rl_policy/control_mode.py) 和 [PicoController](sim2real/rl_policy/controllers/pico.py)。

### 9.6 X 暂停的真实含义

**`X` 暂停时不是保持最后一个人体姿态。**

源码会构造默认站姿参考，同时保留最新参考的 XY 位置和 yaw。再次恢复时，重新建立人体与参考之间的水平位置、朝向对应关系。

实时 backend 不使用 policy 本地的 `paused` 来停止采样；是否暂停由 publisher 输出的参考决定。因此，离线的空格暂停和 PICO 的 `X` 暂停是不同机制。

### 9.7 Publisher 与 policy 不在同一台机器

需要同时修改动作流和按键流地址：

```bash
--motion-zmq-connect tcp://<publisher-ip>:28701 \
--pico-zmq-connect tcp://<publisher-ip>:5592
```

只修改动作地址，会导致动作流能收到，但 `A/B` 模式切换信号收不到。

手部 grip 使用独立的 `5593` 通道，供单独的手部控制程序消费，不属于这里的 G1 身体策略控制闭环。

## 10. PICO 延迟从哪里来

PICO publisher 默认 30 Hz，策略运行在 50 Hz，两者通过 [RealtimeMotionBuffer](sim2real/rl_policy/utils/motion_buffer.py) 衔接。

Buffer 会按时间戳缓存、对齐和插值参考，构造策略需要的动作窗口。

它还要解决一个问题：**策略可能需要未来帧，但实时输入无法提前知道人体未来动作。**

当前实现采用延迟采样：

```text
参考基准时间 = 当前时间 - buffer_delay
buffer_delay = 最大未来步数 × 参考步长 + tolerance
```

当前 SP-Tracking 的窗口是：

```text
future_steps = [-42, -41, ..., 0, ..., 7]
```

在 50 Hz 下，覆盖相对参考基准的：

```text
过去 0.84 秒 → 当前 → 未来 0.14 秒
```

默认 `tolerance=0.04 s`，所以 buffer 主动引入：

```text
7 × 0.02 + 0.04 = 0.18 秒
```

这样，相对参考基准的“未来帧”实际已经在过去收到。

**这 180 ms 只是该配置的参考缓冲延迟，不是端到端总延迟。** 人体追踪、网络传输、重定向、推理和物理响应还会叠加时间。

如果出现“动作能跟，但明显慢半拍”，除了看 ONNX 推理耗时，还应该看模型的 `future_steps`。不同策略的未来参考需求不同，不能把这里的 180 ms 当作所有模型的固定值。

## 11. 常见现象与对应检查点

| 现象 | 优先检查 |
| --- | --- |
| policy 提示 `low state not ready.` | `base_sim` 是否启动、`5590` 状态通道是否连通 |
| 按 `]` 后机器人控制已接管，但离线动作不推进 | 在 policy 终端按空格 |
| Publisher viewer 正常，执行机器人不跟 | policy 模式、参考连接、推理日志及 `low_cmd` 通道 |
| PICO 动作流有数据，但 A/B 不生效 | `--controller pico` 和 `--pico-zmq-connect` |
| 按 X 后机器人参考变成站姿 | 这是 publisher 暂停参考的当前实现 |
| Reset sim 后动作没有从头开始 | 双进程重置独立，需要重置 policy 的参考进度 |
| 开着吊带能站，关闭后不稳 | 吊带提供了额外外力，需要按无吊带状态评估策略 |
| 找不到 Mimic-Lite 模型或本地动作 | 当前示例目录未安装，应使用现有模型或准备对应 artifacts |
| PICO 跟踪明显延迟 | 同时检查未来参考窗口、buffer tolerance 和各环节耗时 |

## 12. 源码阅读顺序

| 顺序 | 文件 | 重点 |
| --- | --- | --- |
| 1 | [tracking.py](sim2real/rl_policy/tracking.py) | 离线与实时参考如何切换、更新和暂停 |
| 2 | [base_policy.py](sim2real/rl_policy/base_policy.py) | 一次推理到发送命令的完整循环 |
| 3 | [policy.yaml](checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml) | 观测、关节顺序、PD 参数和参考窗口 |
| 4 | [bridge.py](sim2real/sim_env/utils/bridge.py) | 命令如何变成 MuJoCo 力矩 |
| 5 | [base_sim.py](sim2real/sim_env/base_sim.py) | 物理步、viewer 和吊带 |
| 6 | [pico_retarget_pub.py](sim2real/teleop/pico_retarget_pub.py) | 人体姿态如何变成参考流、X 如何工作 |
| 7 | [motion_buffer.py](sim2real/rl_policy/utils/motion_buffer.py) | 实时帧如何变成策略需要的时间窗口 |
| 8 | [integrated_sim2sim.py](sim2real/sim_env/integrated_sim2sim.py) | 离线同步执行与轨迹输出 |
