# SONIC release：GR00T PICO 适配

English: [README.md](README.md)。

本目录适配本地 `GR00T-WholeBodyControl` 提交
`087f9ac01d46f6d8e4d0b73c01ae64799f292a38` 默认使用的 SONIC 发布模型。
原 PICO 全身 **POSE** 模式使用 SMPL encoder；可选 G1 encoder 消费重定向后的机器人动作。
这里是默认发布版，不是 low-latency 或 v1.1 模型。

源目录只有代码和配置，没有 SONIC 权重。因此按源项目说明，从官方 `nvidia/GEAR-SONIC`
固定版本 `6733128a3d8a523b1418b06bca3cdf61c8b0987f` 下载 encoder/decoder。
合并时保留原始学习网络和有限标量量化运算，固定模式、补零未启用的输入，每个模式生成
一个完整 ONNX。相邻 `policy.json` 记录源文件哈希、输入布局、模型哈希及数值对照结果。
原代码和模型许可声明见 [LICENSE](LICENSE)。

| 模式 | YAML | 模型输入 | 模型输出 |
| --- | --- | --- | --- |
| PICO 全身 SMPL | `smpl/policy.yaml` | `smpl_input[840]`、`proprioception[930]` | `action[29]`、`token[64]` |
| G1 机器人参考 | `g1/policy.yaml` | `g1_input[640]`、`proprioception[930]` | `action[29]`、`token[64]` |

对外输入输出不带 batch 维，图内自动增减原网络的 batch 维以匹配当前运行时。
保留 IR 7 / opset 13，已验证本机 CPU ORT 1.23.2；未验证 GPU/机载运行。

## 运行 PICO 仿真

在 `mimiclite-deploy` 根目录分别开终端。已有 publisher 需要带 `--publish-smpl` 重启：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py \
  --robot g1 --actual-human-height 1.76 --publish-hz 30 --publish-smpl
```

将身高改成自己的实际米数。如果 MuJoCo 尚未运行，启动：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python sim2real/sim_env/base_sim.py --robot g1
```

退出旧 tracking 进程后，启动新策略：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/rl_policy/tracking.py \
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

按 **A → A+B → X**，依次初始化、进入策略控制、开启实时跟随。
PICO 必须同时开启人体和手柄数据 streaming。本适配沿用当前仓库的手柄控制流程，
没有迁入原项目的 planner、VR 三点模式和按键状态机。

初始暂停会发布官方骨架 FK 生成的根部直立、手臂下垂参考。进入实时跟随后，再按 `X`
会保持最后的 SMPL 人体姿态、手腕和朝向。publisher 并行的 GMR 网页显示机器人参考，
暂停时回到默认站姿，不直接显示 SMPL 策略目标。

SMPL 在 50 Hz 下采样 `[0..9]` 帧：最远前看 180 ms，加 40 ms 缓冲容差，参考延迟共 220 ms。
可选 G1 模式使用 `g1/policy.yaml`、`--motion-backend zmq` 和端口 `28701`，
采样 `[0,5,10,15,20,25,30,35,40,45]`，参考延迟为 940 ms。

## 对齐的源语义与修复

- SMPL 输入依次为全部 10 × 24 × 3 标准人体关节点、10 × 6 相对根部朝向、10 × 6 手腕角。
  G1 输入依次为全部未来关节位置、全部未来关节速度和根部朝向；原 encoder 在内部继续 reshape/拼接。
- 本体观测依次为角速度、关节角减默认值、关节速度、上一轮原始动作、投影重力。
  每项 10 帧，旧帧在前。发布配置显式启用原 C++ 的启动填充和 heading 算法，其他配置仍可使用原默认行为。
- 原 C++ 缺失历史时填零，包括零四元数，其重力计算因此得到 +Z。这一特殊行为已用实际编译的
  原代码验证，而非根据文档猜测。
- 动作顺序、默认角度、PD 和缩放由编译原 `policy_parameters.hpp` 生成，数值记录在
  `source_config.json`。不额外裁剪 action，与源代码的 `default + scale * action` 一致。
- PICO SMPL 手腕改为原项目的肘部 swing/手腕旋转映射；普通 G1 流仍使用原来的重定向关节。
- 修复 SMPL publisher 在初始暂停、尚无实时帧时不发数据的问题，并修复官方回放的关节标签/顺序。
- SMPL buffer 从 30 Hz 重采样到 50 Hz 时，位置和手腕线性插值，旋转使用最短路径 SLERP。

官方 FK 文件 `smpl/human_joints_info.pkl` 来自
`gear_sonic/data/human/human_joints_info.pkl`，SHA256 为
`4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975`。
保留其原生根节点偏移，不将标准关节点重新居中。

## 2026-09-08 验证结果

- 两个模式各 64 组非零输入对照原 encoder → decoder：action 和 token 最大误差均为 **0.0**，重复推理一致。
- 编译原 C++ 的真实历史/旋转函数，对照 25 帧随机状态的部署观测；原始 PICO 预处理也对照源代码，
  骨架误差 ≤3e-6 m，手腕误差 ≤8e-8 rad。
- 根环境测试 **56 项通过、1 项跳过**，另有 2 个部署产物子测试通过。跳过的 publisher 测试需要 GMR，
  已在 teleop 环境通过；该环境的 PICO SMPL 和手柄握力测试共 **11 项通过**。
- 实际 `Tracking.step` 回放 700 步：30 Hz SMPL JSON → 50 Hz 控制，观测/动作均有限，
  命令映射和手腕顺序误差为零。单线程 CPU 单步平均 **4.77 ms**、p99 **5.32 ms**、
  最大 **5.73 ms**，没有超过 20 ms。该回放使用内存 I/O，无物理反馈。

集成 MuJoCo 验证使用随机种子 `20260908`，策略 50 Hz、物理 200 Hz；机器人放到参考第零帧，
初始暂停 2 秒时保持策略运行，随后播放并保持末帧：

| 测试 | 仿真总时长 | 最低 pelvis 高度 | 最大倾角 | 最终水平位移误差 | 末帧保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SMPL 中性站姿 | 10 s | 0.785 m | 4.11° | 0.0037 m | 6.02 s |
| SMPL 官方行走 | 44.1 s | 0.726 m | 10.73° | 0.0972 m | 2.09 s |
| G1 暂停站姿 | 10 s | 0.780 m | 0.79° | 0.0009 m | 6.02 s |
| G1 官方行走 | 44.1 s | 0.728 m | 10.42° | 0.1177 m | 2.09 s |

全部完成且未倒地。官方行走配对数据来自 `walk_forward_amateur_001__A001`，
50 Hz 下为 2002 帧、40.02 秒。SMPL 测试通过仅用于审计的状态适配器注入匹配的人体参考，
正式 `integrated_sim2sim.py` 仍只接受 G1 NPZ 动作。水平误差比较机器人/参考各自初始根坐标系内的
最终相对位移，不代表身体或关节跟踪精度。真人 PICO 和真机稳定性仍需连接设备验证。

完整命令、数据、指标和轨迹保存在本机 `outputs/sonic_adaptation_audit/`。重复已准备的测试：

```bash
env -u ALL_PROXY -u all_proxy \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
  uv run --no-sync python outputs/sonic_adaptation_audit/run_integrated.py --case smpl_walk
```

其他 case 为 `smpl_neutral`、`g1_walk`。移除两个环境变量是为了避免本机 `socks://`
代理格式在解析缓存模型资源时触发错误。

## 重新生成

从上述固定 Hugging Face 版本下载 `model_encoder.onnx`、`model_decoder.onnx`、
`observation_config.yaml` 到 `external/groot_sonic_release/`。导出脚本会先校验精确哈希：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python scripts/export_sonic_release.py
uv run --no-sync python scripts/configure_sonic_release.py \
  --source-repo /home/xingyiwang/workspace/GR00T-WholeBodyControl
uv run --no-sync --with pytest python -m pytest \
  tests/test_sonic_release_observations.py tests/test_sonic_pico_smpl.py \
  tests/test_motion_buffer.py tests/test_sonic_dict_policy.py -q
uv run --project venv/pico --no-sync --with pytest --with joblib python -m pytest \
  tests/test_sonic_pico_smpl.py tests/test_pico_hand_grip.py -q
```
