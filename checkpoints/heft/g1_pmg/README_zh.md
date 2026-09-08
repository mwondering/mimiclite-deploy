# HEFT G1 PMG

English: [README.md](README.md)。

本目录适配本地 `/home/xingyiwang/workspace/motion_tracking` 仓库默认的 G1
PICO 跟踪策略。发布模型位于 `sim2real` 分支的
`sim2real/config/g1/ckpts/G1_PMG/policy.onnx`，提交为
`0d5ba31e33397f3543d350d98b637e26d92f470a`。同时核对了当前训练分支 `main`
（`731c3c41a82183153e25ea302612290e88346177`）的 G1 student 观测定义。
这里使用发布的 PMG 模型；新的训练结果或 Compliance 模型需要单独导出和配置。

`policy.onnx` 是单文件，内含原模型的观测归一化、student adaptor、actor 和全部权重。
四个语义输入在图内拼接，唯一输出 `action` 为确定性的 29 关节 actor 均值，
与原部署实际使用的 action 别名数值一致。`policy.json` 记录来源、源文件哈希和导出对照结果。

## 观测与动作约定

| 输入 | 形状 | 按源代码顺序排列的内容 |
| --- | --- | --- |
| `context` | `[1, 1]` | 启动倒计时，reset 后首帧为 24/25 |
| `motion_command` | `[1, 114]` | 参考根部位移，然后是相对根部旋转 |
| `target_motion` | `[1, 806]` | 绝对参考关节角、参考减当前关节角、根部高度、参考重力 |
| `proprioception` | `[1, 808]` | 角速度、重力、绝对关节角、关节速度、上一轮原始动作的历史 |

参考偏移在 50 Hz 下为 `[0,1,2,3,4,5,6,-1,-2,-4,-8,-12,-16]`。
四类本体历史均选取 `[0,1,2,3,4,8,12,16,20]`，最新帧在前；动作历史为连续 8 帧。
历史以零初始化，四个输入共享一个核心，每次推理只推进一次。reset 后会丢弃通用运行时残留的旧动作。

根部位移使用当前参考根部的完整朝向；相对旋转使用机器人实际朝向，取旋转矩阵前两列，逐列展开。
四元数采用 wxyz，并像原仓库的 SciPy 实现一样先归一化。关节角不减默认站姿。
PICO 流提供实际 body/joint 名称后会刷新索引。

YAML 保留原 controller 的左右交错 29 关节动作顺序、默认角度和 PD 参数。
动作先裁剪到 ±10，再计算 `target = default_joint_pos + action_scale * action`；
手臂缩放为 1.0，其余关节为 0.5。参考和动作均按关节名称映射到 MuJoCo/Unitree 顺序。

## 运行 PICO Sim2Sim

在仓库根目录开三个终端。启动 publisher：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py --robot g1
```

启动 MuJoCo：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python sim2real/sim_env/base_sim.py --robot g1
```

先退出占用命令端口 5591 的旧 tracking 进程，再启动新策略：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/heft/g1_pmg/policy.yaml \
  --inference-backend onnx-cpu \
  --robot-io zmq \
  --motion-backend zmq \
  --motion-zmq-connect tcp://127.0.0.1:28701 \
  --controller pico \
  --pico-zmq-connect tcp://127.0.0.1:5592 \
  --rl-rate 50
```

publisher 起初保持暂停，按 `A` 进入初始姿态，再按 `A+B` 进入策略控制，
人体自然站立时按 `X` 开始实时跟随。现有 publisher 负责手柄按钮和人体动作流。
若显示 `Waiting for XR body data from PICO`，需要先在 PICO 的 XRoboToolkit
连接中开启人体 streaming；加载模型不会自动打开设备 streaming。

本适配沿用当前仓库的 PICO 流程：暂停回到默认站立参考，恢复时对齐水平位置和朝向。
没有复制原 VR 流程的 2 秒启动过渡和停止后保持末帧的行为。原仓库的请求/响应协议也不同，
不要把原 VR server 启动在当前 publisher 的端口上。本策略前看 6 帧需要 120 ms，
加上配置的 40 ms 容差，总缓存延迟为 160 ms。尚未测量真人 PICO 实时效果。

已验证本机 CPU ONNX Runtime。模型保持上游 IR 10 / opset 20；
机载 ORT 1.16 环境使用 GPU 前，需要按适配技能说明做兼容转换。

## 2026-09-08 验证结果

- 64 组有结构的非零输入对照：转换前后 ONNX 动作最大绝对误差 **0.0**，重复推理结果确定。
- 12 项 HEFT 测试通过，包含直接调用原仓库观测代码对照 6 × 53 个控制步
  （容差 2e-6）、历史/reset 和实时名称重排；与 motion-buffer 回归测试合计 **48 项通过**。
- 合成 30 Hz PICO JSON 经过实际 50 Hz `Tracking.step` 路径，四个输入形状正确、动作有限，
  动作到命令的顺序/缩放误差 **0.0**。单线程 CPU 在 180 个样本中单步平均 **1.744 ms**、
  最大 **2.228 ms**。该测试用模拟 I/O 替代了真实机器人接口。

集成 MuJoCo 测试将机器人放在动作第 0 帧，策略在初始 2 秒暂停期间持续运行，
随后播放动作并保持末帧。随机种子 `20260908`，策略 50 Hz，物理 200 Hz，`onnx-cpu`：

| 参考动作 | 仿真总时长 | 最低 pelvis 高度 | 最大 pelvis 倾角 | 最终水平位移误差 | 末帧保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原仓库默认站姿 | 12 s | 0.774 m | 2.47° | 0.0010 m | 2.02 s |
| 当前 PICO 暂停站姿 | 10 s | 0.782 m | 1.06° | 0.0006 m | 6.02 s |
| 原仓库 `walk1_subject1` 前 30 秒 | 34 s | 0.735 m | 8.97° | 0.1655 m | 2.02 s |
| `dance1_subject2_0_3945` 前 30 秒 | 34 s | 0.565 m | 23.57° | 0.9800 m | 2.02 s |

四项均完成且未倒地，但舞蹈片段仍有明显位置漂移。表中误差比较机器人和参考动作各自初始
根部坐标系内的最终位移，不代表关节跟踪精度，也不能替代真人 PICO 稳定性测试。

本机完整命令、日志、JSON 指标和根部轨迹位于 `outputs/heft_adaptation_audit/`。
可用 `uv run --no-sync python outputs/heft_adaptation_audit/run_validation.py`
重复已准备的测试片段。原始 walk NPZ 的根部四元数是 xyzw；测试数据已转换为 wxyz qpos，
保留按名称确定的关节顺序及原生 50 Hz 帧率。

## 重新生成模型包

提取部署分支，无需切换训练仓库的工作分支：

```bash
mkdir -p external/motion_tracking_sim2real
git -C /home/xingyiwang/workspace/motion_tracking archive \
  0d5ba31e33397f3543d350d98b637e26d92f470a sim2real README.md .gitattributes \
  | tar -x -C external/motion_tracking_sim2real
uv run --no-sync python scripts/export_heft_policy.py \
  --source-root external/motion_tracking_sim2real
uv run --no-sync --with pytest python -m pytest \
  tests/test_heft_observations.py tests/test_motion_buffer.py -q
```

直接原仓库数值对照测试需要上述提取目录；独立的历史、布局和 reset 回归测试在没有该目录时仍会运行。
