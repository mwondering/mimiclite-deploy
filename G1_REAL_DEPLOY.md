# MimicLite, SP-Tracking, and HEFT: G1 hardware deployment with PICO

中文：[G1_REAL_DEPLOY_zh.md](G1_REAL_DEPLOY_zh.md).

The common entry point is `python -m scripts.g1.deploy`. It uses G1's 29 joints, a 50 Hz control rate, CPU ONNX inference, and inline RobotIO. Each selected YAML retains its own observations, histories, joint order, default pose, action scales, and PD gains.

This guide uses **one G1 onboard Orin**: the PICO receiver service, retargeting publisher, and policy all run on G1. No additional PC is needed. `XRoboToolkit PC Service` is the application's name; it also has an ARM64 version.

```text
PICO body tracking and controllers
    → G1 LAN IP: ARM64 XRoboToolkit PC Service
        → same G1: PICO publisher
            → 127.0.0.1:28701: G1 joint / body references
            → 127.0.0.1:5592: PICO buttons
                → same G1: policy → inline SDK → motors
```

The headset connects to **G1's reachable LAN IP**. Only the two Python processes communicate through `127.0.0.1`. Root and `venv/pico` are two Python environments on the same G1. This setup does not start a MuJoCo simulation or `real_bridge.py` / `real_bridge_cpp.py`. All three policy families consume G1 references and do not require SONIC's `--publish-smpl` or port `28702`.

## 1. Select a policy

| `--policy` | Deployment configuration |
| --- | --- |
| `mimiclite` | `checkpoints/mimic-lite/roa_9287d8e0/policy.yaml` |
| `sp_tracking` | `checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml` |
| `sp_tracking_0728` | `checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml` |
| `heft` | `checkpoints/heft/g1_pmg/policy.yaml` |

`sp_tracking` selects checkpoint 105000; checkpoint 0728 has a separate alias. Keep each ONNX paired with its own YAML. A custom configuration can use `--policy <family-alias> --policy-config /absolute/path/policy.yaml` and must also pass preflight. A custom configuration cannot be combined with `check --policy all`.

## 2. Prepare code and environments while online

Run all commands on **G1's onboard computer**. The current robot checkout is:

```bash
cd /home/unitree/wxy/mimiclite-deploy
```

The code and complete checkpoint directories listed above must be on this machine, preserving relative paths. Do not copy another machine's `.venv` or `venv/pico/.venv`. Inspect the architecture, JetPack / Ubuntu versions, and interfaces first:

```bash
uname -m
dpkg --print-architecture
cat /etc/nv_tegra_release
cat /etc/os-release
bash .agents/skills/configure-g1-sim2real/scripts/inspect_host.sh "$PWD"
ip -br addr
```

The reported error establishes that this G1 is `arm64`, but does not identify its JetPack / Ubuntu version. Typical JetPack 5 uses L4T R35 / Ubuntu 20.04, and JetPack 6 uses R36 / Ubuntu 22.04. Select the installation branch from the actual output.

### Root environment: policy and robot SDK

Identify **the interface connected to G1's control network** for `--robot-interface`. It may differ from the Wi-Fi interface reaching PICO. Do not assume `eth0`.

The root project uses Python 3.10. If CycloneDDS is missing, install its native libraries using [section 3 of the G1 setup runbook](.agents/skills/configure-g1-sim2real/references/g1-setup-runbook.md#3-cyclonedds-and-shell-environment). Set `CYCLONEDDS_HOME` and `LD_LIBRARY_PATH` to the actual installation.

**Prepare the G1 SDK wheel before installing.** `uv.lock` resolves `unitree-interface` from the local `third_party/wheels/` directory. Wheels are excluded by `.gitignore`, so `git clone` / `git pull` will not provide them. Follow the [runtime artifact download instructions](docs/artifacts.md) and restore this Python 3.10 / Linux aarch64 package to the corresponding path in the G1 checkout:

```text
third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl
```

Confirm the file exists before installing:

```bash
ls -lh third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl
uv sync --extra inference-cpu --extra robot-g1
```

### ARM64 receiver service: select the package for the OS

Use an ARM64 package matching G1's architecture and OS. Follow [Orin teleop setup](docs/getting-started/teleop-onboard-orin.md) for the corresponding dependencies.

**JetPack 5 / Ubuntu 20.04:** Project documentation specifies restoring `third_party/prebuilt/jetpack5-aarch64/` from the [shared runtime artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e). Use `rclone`; see the [artifact download instructions](docs/artifacts.md) for the restored directory layout. You need at least the dedicated deb under `xrobotservice/` and the compatible `xrobot-grpc/` libraries. If downloading an archive, preserve the following paths after extraction:

```bash
ls -lh third_party/prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
sudo apt install -y \
  ./third_party/prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

**JetPack 6:** Choose an ARM64 package compatible with the actual Ubuntu version from the [official Releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0). Official v1.0.0 provides the [standard ARM64 package](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb) and an [ARM64 headless alternative](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb). For example, after downloading the standard package into `external/`:

```bash
sudo apt install -y ./external/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb
```

Install only one version. The official filenames do not specify Ubuntu compatibility and do not establish support for JetPack 5; JP5 should use the dedicated Ubuntu 20.04 package above. Do not apply JP5 deb or gRPC libraries to JP6.

The service command below uses the script provided by the standard package. For the headless alternative, inspect its installed files and startup instructions first. Confirm the service script location after installation:

```bash
dpkg -L roboticsservice | rg 'runService.sh$'
test -f /opt/apps/roboticsservice/runService.sh
```

### Same G1: PICO Python environment and SDK

For a first installation, prepare these two source repositories. If they already exist, inspect their branches and local modifications before proceeding; do not overwrite them:

```bash
mkdir -p external
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git \
  external/XRoboToolkit-PC-Service-Pybind
git clone --branch orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
  external/XRoboToolkit-PC-Service
```

**JetPack 5 only:** Before building, replace upstream gRPC with the compatible `xrobot-grpc` libraries above, retaining a backup. This script stops if the backup already exists instead of overwriting it:

```bash
bash <<'SH'
set -eu
sdk_grpc="external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc"
local_grpc="third_party/prebuilt/jetpack5-aarch64/xrobot-grpc"
test -d "$local_grpc/include"
test -d "$sdk_grpc"
if [ -e "$sdk_grpc.upstream" ]; then
  echo "grpc.upstream exists; inspect the previous compatibility installation first." >&2
  exit 1
fi
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
SH
```

See the [JetPack 5 gRPC guide (Chinese)](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/xrobot-grpc-jetpack5.md) for compatibility-library details. On JetPack 6, skip replacement and retain upstream `linux_aarch64/grpc`.

Create the teleop dependency environment first, then build and install the SDK so its native libraries are available before installing the binding:

```bash
uv sync --project venv/pico --no-install-package xrobotoolkit-sdk
bash scripts/setup/setup_xrobot_pybind.sh --arch aarch64
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --project venv/pico --no-sync python -c "import xrobotoolkit_sdk; print(xrobotoolkit_sdk.__file__)"
```

A working SDK does not need rebuilding. For `GLIBC` / `GLIBCXX` import errors, check the package and gRPC compatibility with the OS; do not substitute x86 files.

The setup script must query `pybind11_DIR` using `uv run --no-sync`. Older scripts using `uv run` trigger synchronization and build the SDK before that path is set, causing `pybind11Config.cmake` lookup to fail. Update the setup script and repeat the sequence above.

### G1: persist offline settings

On G1, create `~/.config/sim2real/` and merge these lines into `~/.config/sim2real/env.sh`, preserving existing native-library settings:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export SIM2REAL_ORT_NUM_THREADS=1
```

If G1 uses the CycloneDDS location from the runbook, also include:

```bash
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Add these lines once to the robot user's `~/.bashrc`:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
[ -f "$HOME/.config/sim2real/env.sh" ] && . "$HOME/.config/sim2real/env.sh"
```

The source line can also be added to `~/.profile`. Explicitly run this in every deployment terminal and noninteractive SSH command:

```bash
source "$HOME/.config/sim2real/env.sh"
```

Do not rely on noninteractive shells automatically evaluating `.bashrc`. The common deployment entry point forces Hugging Face offline mode and ignores HTTP / HTTPS / ALL proxy environment variables within its own process. It does not change the current shell's settings.

## 3. Prepare offline assets and run preflight

Both `policy.yaml` and `policy.onnx` must exist. The G1 MJCF and its meshes must also be cached for the runtime user on G1. Root and `venv/pico` share the cache when run by the same user with the same `HF_HOME` / `HF_HUB_CACHE` settings. While online, pre-cache them in G1's root environment:

```bash
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("elijahgalahad/g1_xmls", revision="main"))
PY
```

This is an asset-download step; restore `HF_HUB_OFFLINE=1` for every policy and teleop command afterward. The two environments under the same user do not need separate downloads. For a completely offline G1, transfer the complete cached `models--elijahgalahad--g1_xmls` directory from another machine that has downloaded it. Its default parent is `~/.cache/huggingface/hub/`; preserve `blobs/`, `refs/`, `snapshots/`, and symlinks. Use the actual cache location if `HF_HOME` / `HF_HUB_CACHE` is configured. Copying a single XML is insufficient.

Verify offline resolution in G1's root environment:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync python - <<'PY'
from mjhub import resolve_asset_reference
print(resolve_asset_reference("hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml"))
PY
```

Run preflight for all four configurations from the repository root. This command does not create hardware RobotIO or send motor commands:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python -m scripts.g1.deploy check \
  --policy all \
  --report outputs/g1_deploy_check.json
```

Preflight checks observation inputs, finite values, ONNX outputs, and joint mappings, and performs actual CPU inference. Passing does not establish hardware standing or live teleoperation stability. This documentation update did not log into, install on, or operate a physical G1; existing offline validation does not replace measurements on this Orin.

## 4. G1 terminals 1 / 2: receive PICO and start the publisher

**Terminal 1, receiver service:** Start one receiver instance on G1 and leave it running. A graphical viewer is not required:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash /opt/apps/roboticsservice/runService.sh
```

Connect PICO and G1 to a mutually reachable LAN. Use `ip -br addr` to identify **G1's address reachable from PICO**, and connect to that address in the headset's XRoboToolkit application. Do not enter `127.0.0.1` on the headset. Wear the leg trackers, complete full-body tracking calibration, enable full-body streaming, and confirm controller data is transmitted too.

**Terminal 2, publisher:** Start from the repository root on the same G1, replacing `1.76` with the operator's actual height in meters:

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

Both ZMQ streams stay on G1; ports `28701` / `5592` do not need LAN exposure. `--no-viewer` disables mjviser to reduce onboard load. After real body data arrives, press X before starting the policy to check live / pause log transitions. Return to natural standing and pause afterward. Temporarily omit `--no-viewer` if you need to inspect the retargeted motion visually.

Persistent `Waiting for XR body data from PICO...` means the ARM64 service, headset connection to G1, or full-body streaming needs attention first. Paused mode continuously publishes a default standing reference, so receiving that stream alone does not prove body tracking is working.

## 5. G1 terminal 3: start the selected policy

Exit previous tracking, bridge, or other controller processes in their terminals so only one low-level controller remains. Inspect processes without changing them:

```bash
pgrep -af 'tracking.py|scripts.g1.deploy|real_bridge|g1_debug_mode'
```

The following `run` command opens the hardware interface and switches to low-level control. Prepare the robot and independent emergency stop according to the established G1 hardware test procedure.

Run in a third terminal on the same G1, replacing `<g1-control-interface>` with the actual motor-control interface:

```bash
cd /home/unitree/wxy/mimiclite-deploy
source "$HOME/.config/sim2real/env.sh"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python -m scripts.g1.deploy run \
  --policy mimiclite \
  --robot-interface <g1-control-interface> \
  --pico-host 127.0.0.1 \
  --startup-timeout 15 \
  --stream-timeout 1.0
```

`--pico-host 127.0.0.1` connects to local motion port `28701` and controller port `5592`, matching both bind addresses in terminal 2. This is separate from the address entered on the PICO headset.

`run` also performs offline preflight for the selected model. Before opening hardware RobotIO / DDS, it waits for both PICO streams and validates that the received G1 reference fields are finite, contain all 29 joints, and have valid quaternions. It exits if startup data does not arrive within `--startup-timeout`. During operation, receiving no new data on either stream for `--stream-timeout` triggers a latched fault. These checks still do not establish freshness of PICO's original tracking data.

To switch models, exit the current policy and rerun the command above, replacing its policy line with one of:

```bash
--policy sp_tracking
# or
--policy sp_tracking_0728
# or
--policy heft
```

The 50 Hz policy and 30 Hz retargeting rates are requested rates. Measure whether all three processes meet them concurrently on the actual Orin. For loop overruns or delay, check CPU load and achieved rates first.

Initialize again after each switch before entering policy mode. To record, append `--record --record-output outputs/g1_pico_run.npz` to the complete `run` command.


## Unitree remote Select software stop

`python -m scripts.g1.deploy run` now also listens for Unitree remote Select by default while PICO handles normal operation. This applies to all three policy families; legacy `tracking.py` / bridge entry points do not include this stop channel. Sync `scripts/g1/deploy.py`, `sim2real/rl_policy/real_tracking.py`, and the new `sim2real/rl_policy/robot_io/select_stop.py`. Existing run commands remain valid.

Exit any running policy process first, then check the button on G1 without motor commands. Replace the control interface and press Select after the prompt:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.g1.deploy remote-check --robot-interface eth0 --timeout 15
```

This only receives DDS remote messages: no motor interface, mode switching, or motor commands. A detected press prints `[PASS] Select received`; timeout without a press prints `[UNVERIFIED]` and finishes without blocking deployment. The listener receives both `rt/wirelesscontroller` and G1 HG `rt/lowstate.wireless_remote`; either source refreshes data and Select from either source latches the stop. Deployment startup does not wait for remote packets or require a Select press.

Select latches a stop: subsequent policy inference is skipped and all motor commands become damping with `Kp=0, Kd=2, dq_target=0, tau_ff=0`. The process keeps running and sending damping. Releasing Select, pressing PICO A/B, or reconnecting cannot clear the latch; exit and restart are required. Cleanup cannot overwrite latched damping with position hold. Only a received Select press triggers this stop; remote silence does not. Listener failure logs a warning and prevents reception of new presses. Existing robot-state and PICO fault monitoring remains in place.

Select is bit 3 of the key word, following the [official Unitree remote parser](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/wireless_controller/wireless_controller.py). A separate process listens on the actual `--robot-interface`, isolated from inline C++ DDS. Motor writes serialize the latch check, and a thread polling about every 10 ms attempts damping during inference. This is not a hardware emergency stop or a guaranteed 10 ms response: process stalls, blocked SDK writes, and motor network loss can prevent delivery. Fresh DDS packets do not establish that the remote's radio link is healthy.

**Damping does not maintain standing balance; the robot can sink or fall.** Use support and independent hardware emergency provisions for the first physical test. Only simulated I/O tests were performed here; actual button reception, stop latency, and mechanical behavior on this G1 remain unverified.

## 6. Take control and enable live following

Keep the publisher paused and the operator standing naturally, then follow this sequence:

```text
Press A → wait about 10 seconds for initialization
→ release A/B completely, then press A+B together → enter policy control
→ confirm stable standing with the paused reference → press X → live following
```

| Button / event | Behavior |
| --- | --- |
| Unitree remote `Select` | Latches damping stop; PICO cannot resume it; restart required |
| `A` | Transition toward the selected policy's initialization pose |
| `A+B` | Reset observation history and enter policy control |
| `B` alone | Enter `zero` mode, using PD control to hold measured joint positions; this is not a torque-off emergency stop |
| `X` | Toggle body following / pause; this guide's G1 stream returns to the default standing reference on pause, retaining horizontal position and heading |
| Runtime fault | If valid joint state is available, latch a joint-position PD hold; address the cause and restart. Restored data does not automatically resume policy control. Exit if valid state has never been obtained |
| `Ctrl+C` | Attempt a joint-position hold before exiting if Select has not latched; otherwise preserve damping before exiting. Restoration of G1's high-level controller is not guaranteed |

Confirm stability with the paused reference before trying slow, small movements. X changes the reference; A/B changes the robot's control mode.

Stream checks confirm recent receipt on the two publisher → policy ZMQ streams. They do not establish freshness of PICO's original tracking data or the low-level state returned by the G1 SDK. A+B requests before initialization finishes are ignored; wait about 10 seconds, release, and press again.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| `package architecture (amd64) does not match system (arm64)` | The package architecture does not match the host. Select an ARM64 service package matching G1's JetPack / Ubuntu as described in section 2 |
| `Failed to download unitree-interface` ending in a local `.whl: No such file or directory` | Restore `third_party/wheels/unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl` as described in section 2, then rerun `uv sync` |
| Missing `unitree_interface` | Install root's `robot-g1` extra on a compatible onboard environment; an ordinary x86 environment does not automatically receive this package |
| DDS interface error / `low state not ready` | Check the control interface, robot connection, low-level state, and SDK; the local motion port is separate from robot state I/O |
| Motion / controller stream startup timeout | Check the local publisher, both binds using `127.0.0.1`, policy `--pico-host 127.0.0.1`, and ports `28701` / `5592` |
| Moving reference but A/B has no effect | Check controller streaming and port `5592`; release buttons before triggering them again |
| Stream timeout during operation | Check PICO, publisher, and network; fix the cause, restart, and repeat initialization |
| `Address already in use ...5591` | Usually an old `tracking.py --robot-io zmq` process remains. This inline entry point does not bind `5591`, but old control processes must still be stopped |
| Missing HF cache | Complete the runtime user's cache and verify the offline resolver and `check`, rather than repeatedly starting hardware processes |
| PICO publisher / resolver reports `Unknown scheme for proxy URL` | Run `unset ALL_PROXY all_proxy` in the deployment terminal and retry. Some HTTP clients parse proxy settings during initialization even in offline mode. The common deployment entry point already ignores HTTP proxies within its own process |
| Unsupported ONNX IR / opset | Check root's CPU ONNX Runtime installation; do not reuse an old Orin GPU ORT environment or arbitrarily edit model versions |
| Shaking only during X-enabled live following | Inspect publisher reference jitter, actual frame rate, and network latency; retain recordings and evaluate each policy's live tracking separately |

See the [G1 setup runbook](.agents/skills/configure-g1-sim2real/references/g1-setup-runbook.md) for installation diagnostics. Each checkpoint directory's `README.md` describes its source and existing simulation results.
