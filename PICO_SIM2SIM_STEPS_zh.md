# PICO 遥操作 Sim2Sim：操作步骤

本文适用于当前 Ubuntu 24.04 电脑和仓库中的 SONIC release SMPL 模型。所有项目命令均在仓库根目录执行：

```bash
cd /home/xingyiwang/workspace/mimiclite-deploy
```

当前 Python 环境、SDK 和 GMR 已安装，且 SDK、GMR、PICO publisher 的导入和 publisher 的 `--help` 已验证。PC Service、PICO 设备连接和该模型的实时遥操作效果仍需按下述步骤实际确认。

该模型来自 `GR00T-WholeBodyControl` 的默认 SONIC 发布版本，保留原 PICO 全身遥操作的 SMPL 模式。模型和官方 `human_joints_info.pkl` 骨架文件已包含在仓库中。
这里使用本仓库的 publisher 和 `A → A+B → X` 控制流程，进入全身 SMPL 跟踪，不启动原项目的 planner 或 VR 三点模式。来源和验证结果见[部署文件说明](checkpoints/sonic/release/README_zh.md)。

运行时需要 **PC Service + 三个 Python 进程**：

```text
PICO 全身追踪
→ XRoboToolkit PC Service
→ PICO publisher：标准 SMPL 骨架与手腕参考
→ Tracking policy：策略推理
→ MuJoCo：物理执行
```

## 1. 安装并启动 XRoboToolkit PC Service

Python SDK 和 PC Service 是两个不同组件。此前安装的是 Python SDK，PC Service 应用需要单独安装；如果已经安装，可以跳过安装命令。

以下安装包对应 Ubuntu 24.04、x86_64：

```bash
gh release download v1.0.0 \
  --repo XR-Robotics/XRoboToolkit-PC-Service \
  --pattern 'XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb' \
  --dir /tmp

sudo apt install \
  /tmp/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

如果安装包已在 `/tmp` 中，无需重复下载。

安装后，从应用菜单打开 **XRoboToolkit / XRobot**，保持运行。同一时间只启动一个 PC Service。

## 2. 连接 PICO，开启全身追踪

1. 让 PICO 和电脑连接到同一个局域网。
2. 戴好头显和腿部 trackers，完成全身追踪校准。
3. 打开 PICO 端的 XRoboToolkit 应用。
4. 连接到这台电脑，开启全身数据 streaming，并确认手柄按键数据也在传输。

如果需要填写电脑 IP，可以运行：

```bash
hostname -I
```

选择电脑连接 PICO 所在网络的局域网地址。

完成后，应先确认 PC Service 能收到设备数据。仅安装 Python SDK 不会自动完成头显连接。

## 3. 终端一：启动 PICO 重定向 publisher

```bash
cd /home/xingyiwang/workspace/mimiclite-deploy

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --robot g1 \
  --actual-human-height 1.76 \
  --publish-hz 30 \
  --publish-smpl
```

将 `1.76` 改为你的实际身高，单位是米。

`--publish-smpl` 必须保留，它在端口 `28702` 发布 SONIC 所需的人体骨架和手腕参考。

打开该终端打印的 **mjviser URL**。这个网页显示并行生成的 GMR 机器人参考，用于检查设备追踪连接；它不直接显示 SONIC SMPL encoder 消费的骨架。

Publisher 默认处于暂停状态：

1. 按一次 PICO 的 **X**，开启实时参考动作。
2. 缓慢抬手、转身，确认网页里的参考机器人会跟随。
3. 验证完成后，先恢复自然站立，再按一次 **X** 保持当前 SMPL 姿态，准备启动控制。

如果提示：

```text
Waiting for XR body data from PICO...
```

优先检查 PC Service、PICO 连接及全身 streaming。

## 4. 终端二：启动 MuJoCo 仿真

```bash
cd /home/xingyiwang/workspace/mimiclite-deploy

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run sim2real/sim_env/base_sim.py --robot g1
```

打开这个终端打印的 **另一个 mjviser URL**，它显示实际物理仿真中的机器人。

两个网页的区别：

| 网页 | 内容 |
| --- | --- |
| Publisher 网页 | 并行生成的 GMR 机器人参考，供检查追踪连接 |
| 仿真网页 | 策略实际控制出来的动作 |

初次启动可以先保持仿真网页中的 `Elastic Band / Enabled` 开启，辅助初始化。

不要同时启动多份 `base_sim.py` 或 tracking 控制进程，以免出现端口占用或控制来源混淆。如果之前还在运行离线 dance/walk tracking，先在对应终端按 `Ctrl+C` 退出它。

## 5. 终端三：启动实时 tracking

```bash
cd /home/xingyiwang/workspace/mimiclite-deploy

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

与离线 dance/walk 播放相比，关键变化是：

```text
motion-backend：npz → smpl_zmq
controller：keyboard → pico
不再指定 motion-path
```

这里使用普通 `tracking.py`，当前 `integrated_sim2sim.py` 不支持 PICO 实时参考。

Publisher 在另一台电脑时，需要同时修改动作流和按键流地址：

```bash
--motion-zmq-connect tcp://<publisher-ip>:28702 \
--pico-zmq-connect tcp://<publisher-ip>:5592
```

只修改动作流地址，可能导致参考动作能收到，但 A/B 模式切换信号收不到。

## 6. 用 PICO 按键接管控制

先保持人体自然站立，按下面的顺序操作：

```text
A
→ 等待仿真机器人靠近初始化姿态
→ 同时按 A+B
→ 进入策略控制
→ 关闭仿真网页中的 Elastic Band
→ 确认机器人能站稳
→ 按 X，开启实时人体动作
```

然后从缓慢抬手、小幅转身等动作开始，观察仿真机器人是否跟随人体；publisher 的 GMR 网页只作辅助检查。

| 按键 | 作用 |
| --- | --- |
| `A` | 初始化姿态 |
| `A+B` | 进入策略控制 |
| `B` 单独按 | 切换到 `zero` 模式，并非关闭所有力矩 |
| `X` | 切换实时动作与暂停参考 |

**SMPL 模式下，X 暂停会保持最后一个人体姿态、手腕参考和朝向，MuJoCo 继续运行。** 首次实时运行前，publisher 用官方 FK 构造直立、双臂下垂的中性参考。并行的 GMR 网页 / G1 流暂停时会回到默认站姿，所以它可能与 SMPL policy 的暂停目标不同。

虚拟吊带会给机器人提供额外外力，因此判断策略是否能够自主站稳和跟踪，需要观察关闭吊带后的表现。

## 7. 判断是否真正跑通

应同时满足：

- Publisher 网页里的参考机器人跟随人体。
- Tracking 没有持续报 `low state not ready` 或推理错误。
- MuJoCo 网页里的执行机器人也跟随动作。
- 关闭吊带后仍能维持稳定。

当前默认频率：

| 环节 | 频率 |
| --- | --- |
| PICO publisher | 30 Hz |
| Tracking policy | 50 Hz |
| MuJoCo 物理步进 | 200 Hz |

Tracking 接收端会根据时间戳插值参考，供 50 Hz 策略使用；当前链路没有自动插值到 90 Hz 的环节。

当前 SONIC SMPL 使用 10 帧、20 ms 间隔的参考，最远帧为 180 ms；加上 40 ms 容差，实时参考缓冲约 **220 ms**，此外还有追踪与执行延迟。

如果要使用 SONIC G1 模式跟踪现有 GMR 机器人参考，可以改为 `checkpoints/sonic/release/g1/policy.yaml` 和 `--motion-backend zmq --motion-zmq-connect tcp://127.0.0.1:28701`。该模式不需要 `--publish-smpl`，最远参考帧为 900 ms，且 X 暂停返回默认站姿。

## 8. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| `Waiting for XR body data from PICO...` | PC Service、PICO 连接和全身 streaming |
| Publisher 网页一直是站姿 | Publisher 默认暂停，按 X 切换到实时模式 |
| `low state not ready.` | MuJoCo 仿真是否启动，状态端口 `5590` 是否连通 |
| 参考机器人动，执行机器人不动 | 是否按 A+B 进入策略控制、tracking 是否有推理错误 |
| SMPL 数据一直未就绪 | Publisher 是否加了 `--publish-smpl`，动作端口是否为 `28702` |
| A/B 按键无效 | `--controller pico` 及按键端口 `5592` 的地址 |
| 网页没有出现 | 打开各进程打印的 URL；tracking 自身不提供网页 |
| 端口被占用 | 是否还运行着旧的仿真、tracking 或 publisher 进程 |
| 关闭吊带后不稳 | 检查参考姿态、初始化及策略跟踪效果 |

`Reset sim` 只重置物理仿真，不会自动同步重置另外两个进程。需要重新开始时，先恢复自然站立并让 publisher 暂停，再重新初始化并进入策略模式。

## 9. 结束运行

在三个 Python 终端分别按 `Ctrl+C`。不再使用 PICO 时，可以退出 PC Service。

## 相关文件

- [Sim2Sim 原理与详细说明](SIM2SIM_GUIDE_zh.md)
- [PICO publisher 源码](sim2real/teleop/pico_retarget_pub.py)
- [Tracking 源码](sim2real/rl_policy/tracking.py)
- [MuJoCo 仿真源码](sim2real/sim_env/base_sim.py)
- [PICO PC 环境文档](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/teleop-x86-64.md)
