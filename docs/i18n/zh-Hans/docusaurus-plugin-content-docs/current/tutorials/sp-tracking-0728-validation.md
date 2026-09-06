---
title: SP-Tracking 0728 验证记录
slug: /tutorials/sp-tracking-0728-validation
---

# SP-Tracking 0728 验证记录

日期：2026-09-06。源目录：`0728_baoshou_waist_dataclean_changedr`，训练步数
22000。环境：根项目，Python 3.10、MuJoCo 3.10.0、ONNX Runtime 1.23.2 CPU。
随机种子：`20260728`。策略频率 50 Hz，物理仿真频率 200 Hz。
源导出及适配模型的哈希见[部署说明](sp-tracking-0728.md)。

## 接口核验

- 源 ONNX 与语义输入 ONNX：20 组随机输入比较，最大及平均动作误差严格为 0。
  模型格式为 ONNX IR 8 / opset 18。
- 29 个关节的名称、默认角度、动作缩放、刚度和阻尼，与源部署文件
  `sim2real/config/g1/tracking_spv5_2.yaml` 完全一致。
- 训练解析正运动学与部署 MuJoCo 雅可比计算：100 组随机 q/dq/gyro，
  特征最大绝对误差 `1.9073486e-6`，平均误差 `5.1618059e-8`。
  此比较调用训练侧 helper API，并使用训练 `g1_tracking_bfm/g1.xml`。
- 26 项针对性测试通过，覆盖历史顺序、共享状态更新、Robot I/O 力矩传递，
  以及动作在缩放和写入下一步历史前完成裁剪。
- 源部署将动作裁剪到 ±10。两条运行路径现在均执行 YAML 中可选的
  `clip_actions`；未配置该字段的策略保留原行为。
- 两种启动模式均通过 `--dry-run` 参数检查，覆盖本地 NPZ 和 Pico/ZMQ 配置。
  CPU ONNX 推理输出 29 维动作。修改的 Python 运行时及适配脚本均通过语法检查。

训练仓库可用时，复现正运动学比较和测试：

```bash
uv run --no-sync python scripts/validate_sp_tracking_kinematics.py \
  --source-repo ../SP_Tracking --trials 100 --seed 20260728
uv run --no-sync --with pytest pytest -q \
  tests/test_sp_tracking_observations.py tests/test_robot_io.py
```

## 集成 sim2sim

两次运行均从动作第 0 帧初始化机器人，初始暂停两秒期间保持策略运行。
使用仓库共享仿真机器人及正常接触／动力学配置，不启用训练随机化。

```bash
uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --headless --run-once \
  --root-trajectory-output outputs/sp_tracking_0728/walk_verified_root.npz

uv run --no-sync python scripts/run_sp_tracking_0728.py sim2sim \
  --motion-path datasets/root90/motions/backward__00__FairySteps.npz \
  --headless --max-runtime-s 20 \
  --root-trajectory-output outputs/sp_tracking_0728/backward_verified_hold_root.npz
```

默认步行 NPZ 包含 7,840 帧 qpos，现有动作加载器将其重采样为 50 Hz 下的
13,066 个策略帧；后退动作包含 401 个策略帧。每个输出 NPZ 保存准确的模型／
动作路径、种子、机器人／参考轨迹、起终点和相对终点误差。
根位移误差分别在机器人与参考轨迹各自的初始根坐标系中计算。

| 指标 | `walk1_subject1` | `backward__00__FairySteps` |
| --- | ---: | ---: |
| 到达的参考末帧 | 13065 / 13065 | 400 / 400 |
| 总仿真时长 | 263.32 s | 20.00 s |
| 末帧保持 | 到末帧退出 | 10.015 s |
| 机器人最低根高度 | 0.626 m | 0.775 m |
| 机器人相对终点位移 (x, y, z)，m | (1.118, 0.569, 0.046) | (−0.953, 0.127, −0.024) |
| 参考相对终点位移 (x, y, z)，m | (0.465, −1.679, 0.019) | (−1.522, 0.153, −0.122) |
| 三维根终点位移误差 | 2.3417 m | 0.5786 m |
| 平面根终点位移误差 | 2.3416 m | 0.5701 m |

后退动作的指标包含末帧保持阶段。两次轨迹数值均为有限值、均到达参考末帧，
均未观察到跌倒。

这些验证用于检查部署链路，不代表全数据集跟踪精度。长步行动作虽完整运行且未跌倒，
仍存在明显的累计位置误差。本次未测试 G1 硬件连接、电机动作、GPU 后端或机载时延；
已准备实机启动命令，但不等于完成实机验证。
