"""Unified G1 deployment: ``python -m scripts.g1.deploy {check,run} --help``."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import socket


ROOT = Path(__file__).resolve().parents[2]
POLICIES = {
    "mimiclite": "checkpoints/mimic-lite/roa_9287d8e0/policy.yaml",
    "sp_tracking": "checkpoints/sp-tracking/spv5_2a_0907_105000/policy.yaml",
    "sp_tracking_0728": "checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml",
    "heft": "checkpoints/heft/g1_pmg/policy.yaml",
}


def _positive(value: str) -> float:
    import math

    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        cmd = sub.add_parser(name, help="Offline validation" if name == "check" else "Control a real G1")
        choices = list(POLICIES) + (["all"] if name == "check" else [])
        cmd.add_argument("--policy", choices=choices, default="all" if name == "check" else "mimiclite")
        cmd.add_argument("--policy-config", type=Path, help="Override the selected profile's YAML")
        if name == "check":
            cmd.add_argument("--report", type=Path, help="Write a JSON validation report")
        else:
            cmd.add_argument("--robot-interface", required=True, help="Actual G1 control network interface")
            cmd.add_argument("--pico-host", default="127.0.0.1", help="Host running pico_retarget_pub.py")
            cmd.add_argument("--motion-zmq-connect", help="Override tcp://PICO_HOST:28701")
            cmd.add_argument("--pico-zmq-connect", help="Override tcp://PICO_HOST:5592")
            cmd.add_argument("--startup-timeout", type=_positive, default=15.0)
            cmd.add_argument("--stream-timeout", type=_positive, default=1.0)
            cmd.add_argument("--record", action="store_true")
            cmd.add_argument("--record-output")
    remote = sub.add_parser("remote-check", help="Read Select packets only; no motor commands or mode switch")
    remote.add_argument("--robot-interface", required=True)
    remote.add_argument("--timeout", type=_positive, default=15.0)
    return parser


def policy_path(args) -> Path:
    path = args.policy_config if args.policy_config is not None else ROOT / POLICIES[args.policy]
    return path.expanduser().resolve()


def configure_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")
    # HF/httpx may parse proxies even when resolving only cached assets. DDS
    # and ZMQ do not need HTTP proxies; leave the parent shell untouched.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def validate_host(interface: str) -> None:
    interfaces = {name for _, name in socket.if_nameindex()}
    if interface not in interfaces or interface == "lo":
        raise ValueError(f"Invalid G1 interface {interface!r}; available: {sorted(interfaces)}")
    for module in ("unitree_interface", "unitree_sdk2py"):
        # Importing unitree_interface here would create DDS too early.
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"Missing {module}. On the G1 host install the root environment with "
                "uv sync --extra inference-cpu --extra robot-g1. "
                "The bundled unitree-interface wheel is Linux aarch64 only."
            )


@contextmanager
def deployment_lock():
    """Exclude other instances of this launcher; legacy DDS writers need stopping too."""
    import fcntl

    path = Path(f"/tmp/mimiclite-g1-deploy-{os.getuid()}.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another G1 deploy run is active for this user") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_robot(args, config: Path) -> None:
    from sim2real.rl_policy.controllers.pico import PicoController
    from sim2real.rl_policy.real_tracking import DeferredRobotIO, RealTracking
    from sim2real.rl_policy.robot_io import create_robot_io
    from sim2real.rl_policy.robot_io.select_stop import SelectStopMonitor, SelectStopRobotIO
    from sim2real.rl_policy.tracking import TrackingArgs

    validate_host(args.robot_interface)
    # Bracket a literal IPv6 address while preserving hostnames and IPv4.
    host = args.pico_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    tracking_args = TrackingArgs(
        policy_config=str(config), robot="g1", rl_rate=50.0,
        inference_backend="onnx-cpu", robot_io="inline",
        robot_interface=args.robot_interface, controller="pico", motion_backend="zmq",
        motion_zmq_connect=args.motion_zmq_connect or f"tcp://{host}:28701",
        pico_zmq_connect=args.pico_zmq_connect or f"tcp://{host}:5592",
        record=args.record, record_output=args.record_output,
    )
    with deployment_lock():
        deferred = DeferredRobotIO()
        controller = PicoController(connect=tracking_args.pico_zmq_connect)
        policy = None
        monitor = None
        try:
            policy = RealTracking(
                tracking_args, robot_io=deferred, controller=controller,
                stream_timeout=args.stream_timeout, startup_timeout=args.startup_timeout,
            )
            policy.wait_for_streams()
            monitor = SelectStopMonitor(args.robot_interface, policy.robot_cfg.domain_id)
            # Only this line releases high-level motion mode and opens real DDS.
            deferred.backend = create_robot_io(
                mode="inline", robot_name="g1", robot_cfg=policy.robot_cfg,
                interface=args.robot_interface,
            )
            deferred.backend = SelectStopRobotIO(deferred.backend, monitor, len(policy.robot_cfg.joint_names))
            deferred.backend.start()
            policy.run()
        finally:
            try:
                if policy is not None:
                    policy.close()
                else:
                    controller.close()
                    deferred.close()
            finally:
                if monitor is not None:
                    monitor.close()


def check_remote(args) -> int:
    import time
    from sim2real.config.robots import get_robot_cfg
    from sim2real.rl_policy.robot_io.select_stop import SelectStopMonitor

    monitor = None
    try:
        monitor = SelectStopMonitor(args.robot_interface, get_robot_cfg("g1").domain_id)
        print(f"Listening on {args.robot_interface}. Press Unitree SELECT to test (read-only, {args.timeout}s).", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if monitor.pressed.is_set():
                print("[PASS] Select received; no RobotIO created and no motor commands sent.", flush=True)
                return 0
            reason = monitor.reason
            if reason is not None:
                raise RuntimeError(reason)
            time.sleep(0.01)
        print("[UNVERIFIED] No Select press observed; check finished without blocking deployment.", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[FAIL] Remote check: {exc}", flush=True)
        return 1
    finally:
        if monitor is not None:
            monitor.close()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "remote-check":
        configure_offline_environment()
        return check_remote(args)
    if args.policy_config is not None and args.policy == "all":
        parser.error("--policy-config requires one explicit --policy")
    configure_offline_environment()
    from sim2real.rl_policy.deploy_validation import check_policy

    reports = []
    names = list(POLICIES) if args.policy == "all" else [args.policy]
    for name in names:
        config = ROOT / POLICIES[name] if args.policy == "all" else policy_path(args)
        try:
            report = {"policy": name, "status": "passed", **check_policy(config)}
            print(f"[PASS] {name}: {report['input_shapes']} -> {report['action_shape']}", flush=True)
        except Exception as exc:
            report = {"policy": name, "status": "failed", "config": str(config), "error": str(exc)}
            print(f"[FAIL] {name}: {exc}", flush=True)
        reports.append(report)
    if args.command == "check" and args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    if any(report["status"] != "passed" for report in reports):
        return 1
    if args.command == "run":
        try:
            run_robot(args, policy_path(args))
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"[FAIL] G1 deployment: {exc}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
