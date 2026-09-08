from __future__ import annotations

from types import SimpleNamespace
import os
import time

import numpy as np
import pytest

from scripts.g1 import deploy
from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.real_tracking import DeferredRobotIO, RealTracking
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.utils.command_sender import ActionManager
from sim2real.rl_policy.utils.state_processor import StateProcessor


def test_offline_launcher_removes_proxy_settings_before_loading_assets(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "socks://127.0.0.1:1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    deploy.configure_offline_environment()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert not any(name.lower() in {"http_proxy", "https_proxy", "all_proxy"} for name in os.environ)


class MemoryIO(RobotIO):
    def __init__(self):
        self.qpos = np.zeros(36, dtype=np.float32)
        self.qpos[3] = 1
        self.qpos[7:] = np.linspace(-0.2, 0.2, 29)
        self.available = True
        self.commands = []

    def read_state(self):
        if not self.available:
            return None
        return RobotState(self.qpos.copy(), np.zeros(35), np.zeros(29), 1)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        self.commands.append(tuple(np.asarray(value).copy() for value in
                                   (q_target, dq_target, tau_ff, kp, kd)))


class Buttons(ControllerBase):
    name = "pico"

    def __init__(self):
        self.mode = None
        self.last_receive_monotonic = time.monotonic()

    def get_control_mode(self):
        mode, self.mode = self.mode, None
        return mode


def runtime():
    """Exercise the real loop and command mapping without an SDK or listener."""
    policy = object.__new__(RealTracking)
    policy.num_dofs = policy.num_actions = 29
    policy.total_inference_cnt = 0
    policy.stream_timeout = 1.0
    policy.startup_timeout = 0.01
    policy.fault_reason = policy._hold_target = policy._state_wait_started = None
    policy._init_completed = False
    policy.robot_io = MemoryIO()
    cfg = get_robot_cfg("g1")
    policy.state_processor = StateProcessor(cfg, policy.robot_io)
    policy.action_manager = ActionManager(cfg, {
        "joint_kp": {".*": 20.0}, "joint_kd": {".*": 1.0},
        "default_joint_pos": {".*": 0.0},
    }, policy.robot_io)
    policy.default_dof_angles = np.zeros(29)
    policy.state_dict = {"control_mode": "zero", "action": np.zeros(29), "paused": True}
    policy.controller = Buttons()
    policy.joystick_controller = policy.pico_controller = policy.keyboard_controller = None
    policy.motion_buffer = SimpleNamespace(last_receive_monotonic=time.monotonic())
    policy.motion_backend = "zmq"
    policy.update = lambda: None
    policy.prepare_obs_for_rl = lambda: ({"policy": np.zeros((1, 29))}, {})
    policy._append_record_frame = lambda **kwargs: None
    policy.reset = lambda: None
    policy.policy = lambda inputs: (np.ones(29), np.full(29, 0.1), inputs)
    return policy


def test_deferred_io_cannot_send_until_hardware_is_attached():
    io = DeferredRobotIO()
    assert io.read_state() is None
    with pytest.raises(RuntimeError, match="not been attached"):
        io.write_command(*(np.zeros(29) for _ in range(5)))
    io.backend = MemoryIO()
    backend = io.backend
    io.write_command(*(np.zeros(29) for _ in range(5)))
    assert len(backend.commands) == 1
    io.close()
    assert io.backend is None


def test_initialization_is_required_and_mode_reset_clears_previous_actions():
    policy = runtime()
    policy.controller.mode = "policy"
    policy.step()
    assert policy.state_dict["control_mode"] == "zero"
    policy.controller.mode = "init"
    policy.step()
    assert policy.init_count == 1
    policy.init_count = 500
    policy.step()
    assert policy._init_completed
    seen_actions = []
    policy.reset = lambda: seen_actions.append(policy.state_dict["action"].copy())
    policy.controller.mode = "policy"
    policy.step()
    np.testing.assert_array_equal(seen_actions[0], np.zeros(29))
    np.testing.assert_allclose(policy.robot_io.commands[-1][0], 0.1)


@pytest.mark.parametrize("stream", ["motion", "controller"])
def test_stream_loss_latches_hold_without_automatically_resuming(stream):
    policy = runtime()
    policy.state_dict["control_mode"] = "policy"
    policy.step()
    holder = policy.motion_buffer if stream == "motion" else policy.controller
    holder.last_receive_monotonic = time.monotonic() - 2
    policy.step()
    expected = policy.robot_io.qpos[7:].copy()
    assert "stream missing/stale" in policy.fault_reason
    np.testing.assert_array_equal(policy.robot_io.commands[-1][0], expected)
    holder.last_receive_monotonic = time.monotonic()
    policy.controller.mode = "policy"
    policy.robot_io.qpos[7:] += 0.1
    policy.step()
    np.testing.assert_array_equal(policy.robot_io.commands[-1][0], expected)
    assert policy.state_dict["control_mode"] == "zero"


@pytest.mark.parametrize("failure", ["exception", "nan_target", "missing_state"])
def test_fault_replaces_last_policy_command_with_valid_hold(failure):
    policy = runtime()
    policy.state_dict["control_mode"] = "policy"
    policy.step()
    if failure == "exception":
        def broken(_inputs):
            raise RuntimeError("ONNX failed")
        policy.policy = broken
    elif failure == "nan_target":
        policy.policy = lambda inputs: (np.zeros(29), np.full(29, np.nan), inputs)
    else:
        policy.robot_io.available = False
    policy.step()
    assert policy.fault_reason is not None
    np.testing.assert_array_equal(policy.robot_io.commands[-1][0], policy.robot_io.qpos[7:])
    assert all(np.isfinite(value).all() for value in policy.robot_io.commands[-1])


def test_no_state_times_out_without_issuing_any_command():
    policy = runtime()
    policy.robot_io.available = False
    policy._state_wait_started = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="No valid G1"):
        policy.step()
    assert not policy.robot_io.commands


def test_model_preflight_failure_never_enters_hardware_runner(monkeypatch, tmp_path):
    import sim2real.rl_policy.deploy_validation as validation

    def invalid(_config):
        raise ValueError("wrong tensor shape")
    monkeypatch.setattr(validation, "check_policy", invalid)
    monkeypatch.setattr(deploy, "run_robot", lambda *_: pytest.fail("hardware runner called"))
    assert deploy.main(["run", "--robot-interface", "fake0", "--policy", "heft"]) == 1


@pytest.mark.parametrize("fail_remote", [True, False])
@pytest.mark.parametrize("fail_wait", [True, False])
def test_robot_attaches_only_after_full_construction_and_both_streams(monkeypatch, fail_wait, fail_remote):
    import sim2real.rl_policy.controllers.pico as pico_module
    import sim2real.rl_policy.real_tracking as real_module
    import sim2real.rl_policy.robot_io as io_module
    import sim2real.rl_policy.robot_io.select_stop as stop_module

    events = []
    class Controller:
        def __init__(self, **kwargs):
            events.append("controller")
        def close(self):
            events.append("controller_closed")
    class Policy:
        def __init__(self, _args, **kwargs):
            events.append("policy_constructed")
            self.robot_cfg = get_robot_cfg("g1")
            self.io = kwargs["robot_io"]
        def wait_for_streams(self):
            events.append("streams_checked")
            if fail_wait:
                raise TimeoutError("controller stream missing")
        def run(self):
            events.append("run")
            assert self.io.backend is not None
        def close(self):
            events.append("closed")
            self.io.close()
    class Monitor:
        reason = None
        def __init__(self, interface, domain):
            assert interface == "test0"
        def wait_until_ready(self, timeout):
            events.append("remote_checked")
            if fail_remote:
                raise TimeoutError("No remote")
        def close(self):
            events.append("monitor_closed")
    monkeypatch.setattr(stop_module, "SelectStopMonitor", Monitor)
    def create(**kwargs):
        events.append("hardware_opened")
        assert kwargs["interface"] == "test0"
        return MemoryIO()
    monkeypatch.setattr(deploy, "validate_host", lambda _: None)
    monkeypatch.setattr(pico_module, "PicoController", Controller)
    monkeypatch.setattr(real_module, "RealTracking", Policy)
    monkeypatch.setattr(io_module, "create_robot_io", create)
    args = deploy.build_parser().parse_args(["run", "--robot-interface", "test0"])
    if fail_wait or fail_remote:
        with pytest.raises(TimeoutError):
            deploy.run_robot(args, deploy.policy_path(args))
        assert "hardware_opened" not in events
    else:
        deploy.run_robot(args, deploy.policy_path(args))
        assert events.index("streams_checked") < events.index("hardware_opened") < events.index("run")
    assert "closed" in events
    if not fail_wait:
        assert events[-1] == "monitor_closed"
    if not fail_wait and not fail_remote:
        assert events.index("remote_checked") < events.index("hardware_opened")


def test_nonfinite_raw_actions_are_rejected_before_clipping(monkeypatch):
    from sim2real.rl_policy import base_policy

    policy = object.__new__(base_policy.BasePolicy)
    policy.policy_config = {"clip_actions": 10.0}
    policy.inference_backend = "onnx-cpu"
    policy.controlled_joint_indices = list(range(29))
    monkeypatch.setattr(base_policy, "build_inference_module", lambda *_:
                        lambda _inputs: {"action": np.full(29, np.inf)})
    policy.setup_policy("unused.onnx")
    with pytest.raises(ValueError, match="finite policy actions"):
        policy.policy({})


@pytest.mark.parametrize("invalid", ["joint_names", "joint_pos", "quaternion"])
def test_received_but_invalid_reference_is_rejected_before_hardware(invalid):
    policy = runtime()
    names = list(get_robot_cfg("g1").joint_names)
    policy.policy_joint_names = names
    policy.motion_config = {"root_body_name": "pelvis"}
    reference = SimpleNamespace(
        joint_pos=np.zeros((1, 1, 29)), joint_vel=np.zeros((1, 1, 29)),
        body_pos_w=np.zeros((1, 1, 1, 3)), body_quat_w=np.array([[[[1., 0, 0, 0]]]]),
        body_lin_vel_w=np.zeros((1, 1, 1, 3)), body_ang_vel_w=np.zeros((1, 1, 1, 3)),
    )
    if invalid == "joint_names":
        names = names[:-1]
    elif invalid == "joint_pos":
        reference.joint_pos[0, 0, 0] = np.nan
    else:
        reference.body_quat_w[:] = 0
    policy.motion_buffer = SimpleNamespace(
        last_receive_monotonic=time.monotonic(), joint_names=names,
        body_names=["pelvis"], get_obs=lambda: reference,
    )
    with pytest.raises(ValueError):
        policy.wait_for_streams()
    assert not policy.robot_io.commands


@pytest.mark.parametrize("profile", list(deploy.POLICIES))
def test_real_loop_with_each_actual_policy_and_memory_io(profile, monkeypatch):
    """Run the deployed loop's mode transitions and histories on the real ONNXs."""
    from sim2real.rl_policy.deploy_validation import _MemoryRobotIO, _SyntheticTracking
    from sim2real.rl_policy.tracking import TrackingArgs

    path = deploy.ROOT / deploy.POLICIES[profile]
    if not path.is_file() or not path.with_suffix(".onnx").is_file():
        pytest.skip("Deployment checkpoint is not installed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("SIM2REAL_ORT_NUM_THREADS", "1")

    class OfflineRealTracking(RealTracking, _SyntheticTracking):
        def _init_motion_backend(self):
            _SyntheticTracking._init_motion_backend(self)
            self.motion_buffer = SimpleNamespace(
                last_receive_monotonic=time.monotonic(), close=lambda: None,
            )

    io = _MemoryRobotIO(np.asarray(get_robot_cfg("g1").default_qpos))
    controller = Buttons()
    policy = OfflineRealTracking(
        TrackingArgs(policy_config=str(path), controller="pico", motion_backend="zmq"),
        robot_io=io, controller=controller,
    )
    try:
        # Align the synthetic measured state with this family's own defaults.
        io.default_qpos[7:] = policy.default_dof_angles
        controller.mode = "init"
        policy.step()
        policy.init_count = 500
        policy.step()
        controller.mode = "policy"
        for frame in range(32):
            controller.last_receive_monotonic = time.monotonic()
            policy.motion_buffer.last_receive_monotonic = time.monotonic()
            policy.total_inference_cnt = io.frame = frame + 2
            policy.step()
            assert policy.fault_reason is None
            assert policy.state_dict["control_mode"] == "policy"
            assert np.isfinite(io.commands[-1]["q_target"]).all()
        assert len(io.commands) == 34
    finally:
        policy.close()


# Select stop tests use fake SDK writes and local threading only, never DDS.
def test_select_bit_latches_across_release_and_other_buttons():
    import threading
    from sim2real.rl_policy.robot_io.select_stop import _receive_keys
    pressed, seen, stamp = threading.Event(), threading.Event(), SimpleNamespace(value=0.)
    _receive_keys(1 << 2, pressed, seen, stamp)  # Start is NOT Select.
    assert seen.is_set() and not pressed.is_set() and stamp.value > 0
    _receive_keys((1 << 3) | (1 << 8), pressed, seen, stamp)
    assert pressed.is_set()
    _receive_keys(0, pressed, seen, stamp)
    assert pressed.is_set()


def stop_guard():
    from sim2real.rl_policy.robot_io.select_stop import SelectStopRobotIO
    io, monitor = MemoryIO(), SimpleNamespace(reason=None)
    return SelectStopRobotIO(io, monitor, 29), io, monitor


def assert_damping(command):
    for vector in command[:4]:
        np.testing.assert_array_equal(vector, np.zeros(29))
    np.testing.assert_array_equal(command[4], np.full(29, 2.0))


def test_select_overrides_normal_fault_hold_and_shutdown_commands():
    guard, io, monitor = stop_guard()
    normal = [np.ones(29) * i for i in range(1, 6)]
    guard.write_command(*normal)
    np.testing.assert_array_equal(io.commands[-1][3], normal[3])
    monitor.reason = "Select"
    guard.write_command(*normal)
    assert_damping(io.commands[-1])
    monitor.reason = None  # Releasing cannot re-enable policy/hold commands.
    guard.write_command(*normal)
    assert guard.emergency_stop_reason == "Select"
    assert_damping(io.commands[-1])
    guard.close()
    assert_damping(io.commands[-1])
    count = len(io.commands)
    guard.close()
    assert len(io.commands) == count


@pytest.mark.parametrize("fault", ["Select", "Wireless controller stream timed out", "Select listener exited"])
def test_stop_precedes_pico_state_and_inference_even_after_policy_fault(fault):
    policy = runtime()
    guard, io, monitor = stop_guard()
    policy.robot_io = guard
    policy.fault_reason = "previous inference fault"
    monitor.reason = fault
    policy.state_processor._prepare_low_state = lambda: pytest.fail("state read after stop")
    policy.process_controllers = lambda: pytest.fail("PICO handled after stop")
    policy.policy = lambda _: pytest.fail("inference after stop")
    policy.step()
    assert_damping(io.commands[-1])
    guard.close()


def test_stop_thread_sends_damping_without_policy_steps():
    import threading
    guard, io, monitor = stop_guard()
    sent = threading.Event()
    original = io.write_command
    def capture(*args):
        original(*args)
        sent.set()
    io.write_command = capture
    guard.start()
    try:
        monitor.reason = "Select during blocked inference"
        assert sent.wait(1.0)
        assert_damping(io.commands[-1])
    finally:
        guard.close()


def test_select_during_sdk_write_finishes_with_damping():
    import threading
    guard, io, monitor = stop_guard()
    entered, release = threading.Event(), threading.Event()
    original = io.write_command
    def slow_write(*args):
        if np.any(args[3]):
            entered.set()
            assert release.wait(1.0)
        original(*args)
    io.write_command = slow_write
    worker = threading.Thread(target=lambda: guard.write_command(*(np.ones(29) for _ in range(5))))
    worker.start()
    try:
        assert entered.wait(1.0)
        monitor.reason = "Select"
        release.set()
        worker.join(1.0)
        assert not worker.is_alive()
        assert_damping(io.commands[-1])
    finally:
        release.set()
        worker.join(1.0)
        guard.close()


@pytest.mark.parametrize("condition", ["pressed", "stale", "dead", "failed", "unseen"])
def test_monitor_fails_closed(condition):
    import threading
    from sim2real.rl_policy.robot_io.select_stop import SelectStopMonitor
    monitor = object.__new__(SelectStopMonitor)
    monitor.timeout = 2.0
    monitor.pressed, monitor.seen, monitor.failed = (threading.Event() for _ in range(3))
    monitor.seen.set()
    monitor.received = SimpleNamespace(value=time.monotonic())
    monitor.process = SimpleNamespace(is_alive=lambda: condition != "dead")
    assert monitor.reason is None if condition != "dead" else monitor.reason is not None
    if condition == "pressed":
        monitor.pressed.set()
    elif condition == "stale":
        monitor.received.value -= 3
    elif condition == "failed":
        monitor.failed.set()
    elif condition == "unseen":
        monitor.seen.clear()
    assert monitor.reason is not None
    with pytest.raises((RuntimeError, TimeoutError)):
        monitor.wait_until_ready(0.0)


@pytest.mark.parametrize("failed", [False, True])
def test_remote_check_never_opens_robot_or_loads_policy(monkeypatch, failed):
    from sim2real.rl_policy.robot_io import select_stop
    import sim2real.rl_policy.deploy_validation as validation
    import sim2real.rl_policy.robot_io as io_module
    events = []
    class Monitor:
        reason = None
        def __init__(self, interface, domain):
            assert interface == "test0"
            self.pressed = SimpleNamespace(is_set=lambda: True)
        def wait_until_ready(self, timeout):
            if failed:
                raise TimeoutError("no controller")
        def close(self):
            events.append("closed")
    monkeypatch.setattr(select_stop, "SelectStopMonitor", Monitor)
    monkeypatch.setattr(validation, "check_policy", lambda *_: pytest.fail("model loaded"))
    monkeypatch.setattr(io_module, "create_robot_io", lambda **_: pytest.fail("hardware opened"))
    assert deploy.main(["remote-check", "--robot-interface", "test0"]) == int(failed)
    assert events == ["closed"]
