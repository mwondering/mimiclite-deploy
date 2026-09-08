# MimicLite、SP-Tracking、HEFT：G1 真机 PICO 部署

English: [G1_REAL_DEPLOY.md](G1_REAL_DEPLOY.md)。

统一入口是 `python -m scripts.g1.deploy`。它使用 G1 29 关节、50 Hz、CPU ONNX 推理和 inline RobotIO，按所选 YAML 保留各策略的观测、历史、关节顺序、默认姿态、动作缩放与 PD 参数。

本文采用 **单 G1 机载 Orin 方案**：PICO 接收服务、重定向 publisher 和策略全部运行在 G1 上，不需要另一台 PC。`XRoboToolkit PC Service` 是应用名称，也有 ARM64 版本，并非只能在 x86 PC 上运行。

```text
PICO 全身追踪与手柄
    → G1 的局域网 IP：ARM64 XRoboToolkit PC Service
        → 同一台 G1：PICO publisher
            → 127.0.0.1:28701：G1 关节 / 身体参考
            → 127.0.0.1:5592：PICO 按钮
                → 同一台 G1：策略 → inline SDK → 电机
```

头显连接 **G1 可达的局域网 IP**；两个 Python 进程之间才使用 `127.0.0.1`。root 和 `venv/pico` 是同一台 G1 上的两个 Python 环境。这条链路不启动 MuJoCo 仿真或 `real_bridge.py` / `real_bridge_cpp.py`。三类策略都消费 G1 参考，不需要 SONIC 的 `--publish-smpl` 或端口 `28702`。

## 1. 选择策略

| `--policy` | 部署配置 |
| --- | --- |
| `mimiclite` | `checkpoints/mimic-lite/roa_9287d8e0/policy.yaml` |
| `sp_tracking` | `checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml` |
| `sp_tracking_0728` | `checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml` |
| `heft` | `checkpoints/heft/g1_pmg/policy.yaml` |

`sp_tracking` 明确指向 105000，0728 使用单独的别名。不要只替换 ONNX 而复用另一策略的 YAML。自定义配置可使用 `--policy <所属别名> --policy-config /absolute/path/policy.yaml`，仍需通过预检；`check --policy all` 不能同时指定自定义配置。

## 2. 准备环境和代码（联网安装阶段）

以下命令均在 **G1 机载电脑** 执行。当前机器人 checkout 路径是：

```bash
cd /home/unitree/wxy/mimiclite-deploy
```

代码和上表完整 checkpoint 目录须在这台机器上，保留相对路径；不要复制其他机器的 `.venv` 或 `venv/pico/.venv`。先检查架构、JetPack / Ubuntu 和网卡：

```bash
uname -m
dpkg --print-architecture
cat /etc/nv_tegra_release
cat /etc/os-release
bash .agents/skills/configure-g1-sim2real/scripts/inspect_host.sh "$PWD"
ip -br addr
```

当前报错已证实 G1 是 `arm64`，但不能据此判断 JetPack / Ubuntu 版本。典型 JetPack 5 是 L4T R35 / Ubuntu 20.04，JetPack 6 是 R36 / Ubuntu 22.04；按实际输出选择下面的安装分支。

### root 环境：策略与机器人 SDK

确定 **连接 G1 控制网络的网卡名**，后面填入 `--robot-interface`；它与连接 PICO 的 Wi-Fi 网卡可能不同，不默认使用 `eth0`。

root 使用 Python 3.10。若机器缺少 CycloneDDS，先按 [G1 安装手册第 3 节](.agents/skills/configure-g1-sim2real/references/g1-setup-runbook.md#3-cyclonedds-and-shell-environment) 安装原生库，并让 `CYCLONEDDS_HOME`、`LD_LIBRARY_PATH` 指向实际安装位置。

**安装前还需补齐 G1 SDK wheel。** `uv.lock` 固定从本地 `third_party/wheels/` 安装 `unitree-interface`；wheel 被 `.gitignore` 排除，`git clone` / `git pull` 不会带来它。按[运行依赖下载说明](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/artifacts.md)获取部署依赖，将下列 Python 3.10 / Linux aarch64 安装包恢复到 G1 仓库的对应位置：

```text
third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl
```

确认文件存在后再安装：

```bash
ls -lh third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl
uv sync --extra inference-cpu --extra robot-g1
```

### ARM64 接收服务：按系统版本选择安装包

使用匹配 G1 架构和系统版本的 ARM64 包，按[Orin teleop 安装说明](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/teleop-onboard-orin.md)确认对应依赖。

**JetPack 5 / Ubuntu 20.04：** 项目文档指定从[共享运行依赖](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)恢复 `third_party/prebuilt/jetpack5-aarch64/`。使用 `rclone` 获取，恢复目录布局见[运行依赖下载说明](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/artifacts.md)。至少需要 `xrobotservice/` 下的专用 deb 和 `xrobot-grpc/` 兼容库；若下载的是压缩包，解压后须保留以下相对路径：

```bash
ls -lh third_party/prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
sudo apt install -y \
  ./third_party/prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

**JetPack 6：** 从[官方 Releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0)选择与实际 Ubuntu 版本兼容的 ARM64 包。官方 v1.0.0 提供 [ARM64 常规版](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb)和 [ARM64 无界面版](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb)。例如将常规版下载到 `external/` 后：

```bash
sudo apt install -y ./external/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb
```

只安装一个版本。官方文件名未标 Ubuntu 版本，不能据此推断适用于 JetPack 5；JP5 应使用上面的 Ubuntu 20.04 专用包。不要在 JP6 上套用 JP5 的 deb 或 gRPC 库。

后面的服务启动命令按常规包提供的脚本编写；若选择无界面包，先检查其已安装文件和启动说明。安装后确认服务脚本路径：

```bash
dpkg -L roboticsservice | rg 'runService.sh$'
test -f /opt/apps/roboticsservice/runService.sh
```

### 同一台 G1：PICO Python 环境与 SDK

首次安装时准备以下两个源码仓库；已经存在时先检查已有分支和改动，不要覆盖：

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone --branch orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

**仅 JetPack 5**：在构建之前用上述 `xrobot-grpc` 兼容库替换上游 gRPC，保留原目录备份。下面脚本在备份已存在时停止，避免覆盖它：

```bash
bash <<'SH'
set -eu
sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
local_grpc="third_party/prebuilt/jetpack5-aarch64/xrobot-grpc"
test -d "$local_grpc/include"
test -d "$sdk_grpc"
if [ -e "$sdk_grpc.upstream" ]; then
  echo "grpc.upstream 已存在；先检查已有兼容库安装，不重复替换。" >&2
  exit 1
fi
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
SH
```

详细兼容库说明见 [JetPack 5 gRPC 指南](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/xrobot-grpc-jetpack5.md)。JetPack 6 跳过替换，保留上游 `linux_aarch64/grpc`。

先创建 teleop 依赖环境，再构建并安装 SDK，避免在原生库准备好之前安装绑定：

```bash
uv sync --project venv/pico --no-install-package xrobotoolkit-sdk
bash scripts/setup/setup_xrobot_pybind.sh --arch aarch64
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico --no-sync python -c "import xrobotoolkit_sdk; print(xrobotoolkit_sdk.__file__)"
```

已有可用 SDK 无需重复构建。若导入出现 `GLIBC` / `GLIBCXX` 错误，先核对包和 gRPC 是否匹配系统，不要改用 x86 文件。

安装脚本查询 `pybind11_DIR` 时必须使用 `uv run --no-sync`。旧脚本若使用 `uv run`，会在路径设置之前自动同步并提前构建 SDK，导致 `pybind11Config.cmake` 找不到。更新安装脚本后按上面的顺序重新执行。

### G1：持久化离线设置

在 G1 上创建 `~/.config/sim2real/`，将以下内容合并到 `~/.config/sim2real/env.sh`；保留已有的原生库设置：

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS=1
```

G1 若使用安装手册的 CycloneDDS 路径，文件还应包含：

```bash
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
```

在机器人用户的 `~/.bashrc` 添加一次以下内容：

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
[ -f "$HOME/.config/sim2real/env.sh" ] && . "$HOME/.config/sim2real/env.sh"
```

也可把最后一行加入 `~/.profile`。每个部署终端、SSH 非交互命令都显式执行：

```bash
source "$HOME/.config/sim2real/env.sh"
```

不要依赖非交互 shell 自动执行 `.bashrc`。统一部署入口强制启用 Hugging Face 离线模式，并忽略其自身进程内的 HTTP / HTTPS / ALL 代理环境变量；不会修改当前 shell 的设置。

## 3. 准备离线资产并预检

`policy.yaml` 和 `policy.onnx` 均须存在。G1 MJCF 及其网格还需在 G1 运行用户的 Hugging Face 缓存中。root 和 `venv/pico` 使用同一用户、相同 `HF_HOME` / `HF_HUB_CACHE` 时共享缓存。联网时可在 G1 root 环境预缓存：

```bash
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("elijahgalahad/g1_xmls", revision="main"))
PY
```

这是资产下载步骤，完成后所有策略与 teleop 命令恢复 `HF_HUB_OFFLINE=1`。同一用户的两个环境无需重复下载。完全离线的 G1 可接收其他已联网机器缓存的 `models--elijahgalahad--g1_xmls` 整个目录：默认位于 `~/.cache/huggingface/hub/`，保留 `blobs/`、`refs/`、`snapshots/` 和符号链接；设置过 `HF_HOME` / `HF_HUB_CACHE` 时使用实际目录。只复制一个 XML 不够。

在 G1 root 环境检查离线解析：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python - <<'PY'
from mjhub import resolve_asset_reference
print(resolve_asset_reference("hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml"))
PY
```

在仓库根目录运行四个配置的预检；此命令不创建真机 RobotIO，也不发送电机命令：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python -m scripts.g1.deploy check \
  --policy all \
  --report outputs/g1_deploy_check.json
```

预检核对观测输入、有限数值、ONNX 输出与关节映射，并执行实际 CPU 推理。成功不等同于真机站立或真人遥操作稳定；本次文档修订未登录、安装或操作真实 G1，现有离线验证也不能替代这台 Orin 上的实测。

## 4. G1 终端 1 / 2：接收 PICO 并启动 publisher

**终端 1，服务进程：** 在 G1 启动一个接收服务并保持运行，无需图形查看器：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash /opt/apps/roboticsservice/runService.sh
```

PICO 与 G1 接入能互相访问的局域网。用 `ip -br addr` 找到 **PICO 可访问的 G1 地址**，在头显 XRoboToolkit 应用里连接这个地址；不要给头显填 `127.0.0.1`。佩戴腿部 trackers，完成全身追踪校准，开启全身 tracking streaming，并确认手柄数据也在发送。

**终端 2，publisher：** 在同一台 G1 的仓库根目录启动，将 `1.76` 改为实际身高（米）：

```bash
cd /home/unitree/wxy/mimiclite-deploy
source "$HOME/.config/sim2real/env.sh"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico --no-sync python sim2real/teleop/pico_retarget_pub.py \
  --robot g1 \
  --actual-human-height 1.76 \
  --publish-hz 30 \
  --no-viewer \
  --bind tcp://127.0.0.1:28701 \
  --controller-bind tcp://127.0.0.1:5592
```

两个 ZMQ 流只在 G1 本机传输，无需向局域网开放 `28701` / `5592`。`--no-viewer` 关闭 mjviser，可减少机载负载。收到真实人体数据后，可在尚未启动策略时按 X 检查 live / pause 日志切换；随后恢复自然站立并回到暂停模式。需要看重定向动作时可临时去掉 `--no-viewer`。

`Waiting for XR body data from PICO...` 持续出现时，应先处理 ARM64 服务、头显到 G1 的连接和全身 streaming。暂停模式能持续发布默认站姿，并不能证明已经收到人体数据。

## 5. G1 终端 3：启动所选策略

先在旧进程对应的终端退出 tracking、bridge 或其他控制器，确保只有一个低层控制来源。可先只读查看：

```bash
pgrep -af 'tracking.py|scripts.g1.deploy|real_bridge|g1_debug_mode'
```

下面的 `run` 会打开真机接口，并切换到低层控制。操作前按已有的 G1 真机测试流程准备机器人和独立急停。

在同一台 G1 的第三个终端运行，将 `<G1控制网卡>` 替换为实际电机控制网卡：

```bash
cd /home/unitree/wxy/mimiclite-deploy
source "$HOME/.config/sim2real/env.sh"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python -m scripts.g1.deploy run \
  --policy mimiclite \
  --robot-interface <G1控制网卡> \
  --pico-host 127.0.0.1 \
  --startup-timeout 15 \
  --stream-timeout 1.0
```

`--pico-host 127.0.0.1` 同时连接本机的动作端口 `28701` 与按键端口 `5592`，与终端 2 的两个 bind 地址对应。这不是 PICO 头显的连接地址。

`run` 也会先对当前模型执行离线预检，收到两路 PICO 数据，并验证实际 G1 参考字段数值有限、29 关节完整及四元数有效后，才打开真机 RobotIO / DDS；等待数据超过 `--startup-timeout` 会退出。运行中任一路超过 `--stream-timeout` 未收到新数据，会触发故障锁存。这些检查仍不证明 PICO 原始追踪数据正在更新。

换模型时，退出当前策略后重新运行上面的命令，只替换相应一行：

```bash
--policy sp_tracking
# 或
--policy sp_tracking_0728
# 或
--policy heft
```

50 Hz 策略与 30 Hz 重定向是请求频率，三进程并行是否达到实时要求需在具体 Orin 上测量；出现循环超时或延迟时先检查 CPU 负载和实际频率。

每次切换重新初始化，再进入策略模式。需要记录时，在完整 `run` 命令后追加 `--record --record-output outputs/g1_pico_run.npz`。

## 6. 接管与实时跟随

保持 publisher 暂停、人体自然站立，依次操作：

```text
按 A → 等待约 10 秒初始化
→ 完全释放 A/B，再同时按 A+B → 进入策略控制
→ 确认暂停参考下能够站稳 → 按 X → 开始实时人体跟随
```

| 按键 / 事件 | 行为 |
| --- | --- |
| `A` | 向所选策略的初始化姿态过渡 |
| `A+B` | 重置观测历史并进入策略控制 |
| `B` 单独按 | `zero` 模式，以 PD 控制保持测得的关节位置；不是断力矩急停 |
| `X` | 切换人体跟随 / 暂停；本指南的 G1 流暂停会回到默认站姿参考，保留当前水平位置与朝向 |
| 运行故障 | 已有有效关节状态时锁存关节位置 PD 保持；需处理原因并重启，恢复数据不会自动重新接管。尚未取得有效状态时退出 |
| `Ctrl+C` | 尝试发送最后已知关节位置保持命令后退出；不是硬件急停，也不保证自动恢复 G1 高层控制 |

先在暂停参考下确认稳定，再测试缓慢抬手等小幅动作。X 控制参考，A/B 控制机器人模式，两者作用不同。

断流检查确认的是 publisher → policy 两路 ZMQ 最近收到数据，不保证 PICO 原始追踪数据或 G1 SDK 返回的低层状态仍在更新。初始化完成前的 A+B 请求会被忽略，等待约 10 秒后释放并重新按下。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| `package architecture (amd64) does not match system (arm64)` | 安装包与主机架构不符。按第 2 节选择匹配 G1 JetPack / Ubuntu 的 ARM64 服务包 |
| `Failed to download unitree-interface`，末尾为本地 `.whl: No such file or directory` | 缺少部署 wheel；按第 2 节补齐 `third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl`，再运行 `uv sync` |
| `unitree_interface` 缺失 | 在兼容的机载环境安装 root `robot-g1` extra；普通 x86 环境不会自动获得此包 |
| DDS interface / `low state not ready` | 核对控制网卡、机器人连接、低层状态与 SDK；本机动作端口不是机器人状态接口 |
| 动作流 / 手柄流启动超时 | 确认本机 publisher 在运行，两个 bind 均为 `127.0.0.1`，策略 `--pico-host 127.0.0.1`，端口为 `28701` / `5592` |
| 动作参考在动，A/B 无效 | 核对手柄 streaming 和 `5592`，并先释放按键再触发 |
| 运行中 stream 超时 | 检查 PICO、publisher 和网络，处理原因后重新启动并执行初始化流程 |
| `Address already in use ...5591` | 常见于仍在运行旧 `tracking.py --robot-io zmq`；本入口用 inline，不绑定 `5591`，旧控制进程仍应退出 |
| HF 缓存缺失 | 先补全运行用户的缓存，用离线 resolver 和 `check` 验证，避免反复启动真机进程 |
| PICO publisher / resolver 报 `Unknown scheme for proxy URL` | 在部署终端执行 `unset ALL_PROXY all_proxy` 后重试；某些 HTTP 客户端即使离线也会在初始化时解析代理。统一部署入口已在自身进程忽略 HTTP 代理 |
| ONNX IR / opset 不支持 | 核对 root CPU ONNX Runtime 安装；不要直接套用旧 Orin GPU ORT 环境或随意改模型版本 |
| 只有 X 实时跟随时抖动 | 先比较 publisher 参考是否抖动、实际帧率和网络延迟；保留记录，分别检查三类策略的实时跟踪表现 |

通用安装诊断见 [G1 安装手册](.agents/skills/configure-g1-sim2real/references/g1-setup-runbook.md)。策略来源与已有仿真结果见各 checkpoint 目录的 `README_zh.md`。
