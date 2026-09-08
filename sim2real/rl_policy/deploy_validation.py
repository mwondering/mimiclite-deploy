"""Offline G1 deployment checks using the actual observations and CPU policy.

No transport, controller listener, or hardware SDK is constructed here. Synthetic
states exercise tensor contracts and joint mapping; this is not a stability test.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.passive import PassiveController
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.rl_policy.utils.motion_buffer import _default_frame_from_robot_cfg
from sim2real.utils.strings import resolve_matching_names_values


def _joint_values(config: dict[str, Any], key: str, names: list[str]) -> np.ndarray:
    """Resolve runtime regex maps, requiring full coverage for gains and scales."""
    value = config[key]
    if isinstance(value, dict):
        indices, _, values = resolve_matching_names_values(
            value, names, preserve_order=True, strict=True
        )
        missing = [name for index, name in enumerate(names) if index not in indices]
        if missing and key != "default_joint_pos":
            raise ValueError(f"{key} is missing joints: {missing}")
        # BasePolicy defines unspecified default positions as zero. MimicLite's
        # official YAML intentionally lists only its nonzero default joints.
        result = np.zeros(len(names), dtype=np.float64)
        result[indices] = values
    elif key == "action_scale" and isinstance(value, (float, list)):
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.full(len(names), float(result))
    else:
        raise ValueError(f"{key} has an unsupported runtime value type: {type(value).__name__}")
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise ValueError(f"{key} must provide {len(names)} finite scalar values")
    if key in {"joint_kp", "joint_kd"} and np.any(result < 0):
        raise ValueError(f"{key} must be non-negative")
    if key == "action_scale" and np.any(result <= 0):
        raise ValueError("action_scale must be positive")
    return result


def _validate_config(config: dict[str, Any]) -> dict[str, np.ndarray]:
    canonical_names = list(get_robot_cfg("g1").joint_names)
    policy_names = config.get("policy_joint_names", [])
    if (
        len(policy_names) != 29
        or len(set(policy_names)) != 29
        or set(policy_names) != set(canonical_names)
    ):
        raise ValueError("policy_joint_names must contain each of the 29 G1 joints exactly once")
    simulation_names = config.get("joint_names_simulation", [])
    if (
        len(simulation_names) != 29
        or len(set(simulation_names)) != 29
        or set(simulation_names) != set(canonical_names)
    ):
        raise ValueError("joint_names_simulation must contain each of the 29 G1 joints exactly once")
    if not isinstance(config.get("observation"), dict) or not config["observation"]:
        raise ValueError("observation must contain the policy's ONNX input groups")
    result = {
        key: _joint_values(config, key, canonical_names)
        for key in ("default_joint_pos", "joint_kp", "joint_kd")
    }
    result["action_scale"] = _joint_values(config, "action_scale", policy_names)
    return result


class _MemoryRobotIO(RobotIO):
    """A command sink with deterministic measured state, not a dynamics simulator."""

    def __init__(self, default_qpos: np.ndarray) -> None:
        self.default_qpos = np.asarray(default_qpos, dtype=np.float32)
        self.frame = 0
        self.commands: list[dict[str, np.ndarray]] = []

    def read_state(self) -> RobotState:
        phase = self.frame * 0.02
        joint_phase = phase + np.arange(29, dtype=np.float32) * 0.11
        qpos = self.default_qpos.copy()
        qpos[7:] += 0.005 * np.sin(joint_phase)
        # A small, valid IMU rotation and corresponding angular velocity.
        roll = 0.01 * np.sin(phase)
        qpos[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0, 0]
        qvel = np.zeros(35, dtype=np.float32)
        qvel[3] = 0.01 * np.cos(phase)
        qvel[6:] = 0.005 * np.cos(joint_phase)
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            joint_torque=(0.1 * np.sin(joint_phase)).astype(np.float32),
            tick=self.frame + 1,
        )

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        command = {}
        for name, value in {
            "q_target": q_target, "dq_target": dq_target, "tau_ff": tau_ff,
            "kp": kp, "kd": kd,
        }.items():
            array = np.asarray(value)
            if array.shape != (29,) or not np.isfinite(array).all():
                raise ValueError(f"Motor command {name} must have 29 finite values")
            command[name] = array.copy()
        self.commands.append(command)


class _SyntheticTracking(Tracking):
    def _init_motion_backend(self) -> None:
        self.motion_config = dict(self.policy_config["motion"])
        self.motion_backend = "synthetic"
        self.motion_future_steps = np.asarray(self.motion_config["future_steps"], dtype=int)
        if self.motion_future_steps.ndim != 1 or not self.motion_future_steps.size:
            raise ValueError("motion.future_steps must be a non-empty 1D sequence")
        self.motion_joint_names = list(self.robot_cfg.joint_names)
        self.motion_body_names = list(self.robot_cfg.body_names)
        default_qpos = np.asarray(self.robot_cfg.default_qpos).copy()
        default_qpos[7:] = self.default_dof_angles
        # Use the policy's default joints for the same FK used by the live buffer.
        frame = _default_frame_from_robot_cfg(
            replace(self.robot_cfg, default_qpos=tuple(default_qpos)),
            joint_names=self.motion_joint_names,
            body_names=self.motion_body_names,
        )
        if frame is None:
            raise RuntimeError(
                "Cannot construct G1 reference FK from the cached MJCF; "
                "prepare the robot assets before offline deployment validation"
            )
        self._reference_frame = frame
        self._update_motion_data()

    def _update_motion_data(self, *, paused: bool = False) -> None:
        joint_pos, body_pos_w, body_quat_w = self._reference_frame
        count = len(self.motion_future_steps)
        frame = int(getattr(self, "total_inference_cnt", 0))
        times = (frame + self.motion_future_steps) * self.rl_dt
        positions = np.broadcast_to(body_pos_w, (1, count, *body_pos_w.shape)).copy()
        velocities = np.zeros_like(positions)
        # A 5 mm root translation also exercises history/future reference indexing.
        positions[0, :, :, 0] += 0.005 * np.sin(times)[:, None]
        velocities[0, :, :, 0] = 0.005 * np.cos(times)[:, None]
        self.motion_data = MotionData(
            motion_id=np.zeros((1, count), dtype=np.int64),
            step=self.motion_future_steps.reshape(1, -1),
            timestamps_ns=(times * 1e9).astype(np.int64).reshape(1, -1),
            joint_pos=np.broadcast_to(joint_pos, (1, count, 29)).copy(),
            joint_vel=np.zeros((1, count, 29), dtype=np.float32),
            body_pos_w=positions,
            body_lin_vel_w=velocities,
            body_quat_w=np.broadcast_to(
                body_quat_w, (1, count, *body_quat_w.shape)
            ).copy(),
            body_ang_vel_w=np.zeros_like(positions),
        )


def _check_input_contract(runtime, observations: dict[str, np.ndarray]) -> None:
    if set(runtime.in_keys) != set(observations):
        raise ValueError(
            f"ONNX input groups {runtime.in_keys} do not match observations {list(observations)}"
        )
    if len(runtime.in_keys) != len(runtime.ort_session.get_inputs()):
        raise ValueError("ONNX metadata input count does not match graph input count")
    if len(runtime.out_keys) != len(runtime.ort_session.get_outputs()):
        raise ValueError("ONNX metadata output count does not match graph output count")
    for key, expected, binding in zip(
        runtime.in_keys, runtime.input_shapes, runtime.ort_session.get_inputs()
    ):
        normalized_name = binding.name.removesuffix("_orig").removeprefix("next_")
        if key != normalized_name:
            raise ValueError(
                f"ONNX metadata input {key} disagrees with graph binding {binding.name}"
            )
        value = np.asarray(observations[key])
        accepted = tuple(value.shape) == tuple(expected) or (
            value.ndim > 0 and value.shape[0] == 1 and tuple(value.shape[1:]) == tuple(expected)
        )
        if not accepted:
            raise ValueError(f"Observation {key} shape {value.shape} does not match ONNX {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"Observation {key} contains non-finite values")


def check_policy(policy_config: str | Path, *, steps: int = 32) -> dict[str, Any]:
    """Validate a G1 policy on synthetic data without constructing live transports.

    The caller should set HF_HUB_OFFLINE=1 and HF_HUB_DISABLE_TELEMETRY=1 before
    starting Python, as for deployment. All model/FK assets must already be local.
    Raises on incompatible configuration, invalid tensors, or inference failure.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    config_path = Path(policy_config).expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Policy YAML must be a mapping")
    values = _validate_config(config)
    canonical_names = list(get_robot_cfg("g1").joint_names)
    default_indices, _, _ = resolve_matching_names_values(
        config["default_joint_pos"], canonical_names, strict=True
    )
    implicit_zero_joints = [
        name for index, name in enumerate(canonical_names) if index not in default_indices
    ]
    configured_model = config.get("model_path")
    model_path = (
        config_path.with_suffix(".onnx")
        if configured_model is None
        else config_path.parent / str(configured_model)
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Policy ONNX is missing: {model_path}")

    robot_cfg = get_robot_cfg("g1")
    default_qpos = np.asarray(robot_cfg.default_qpos).copy()
    default_qpos[7:] = values["default_joint_pos"]
    robot_io = _MemoryRobotIO(default_qpos)
    controller = PassiveController()
    policy = None
    try:
        policy = _SyntheticTracking(
            TrackingArgs(
                policy_config=str(config_path), robot="g1", rl_rate=50,
                inference_backend="onnx-cpu", robot_io="inline", controller="pico",
                motion_backend="zmq",
            ),
            robot_io=robot_io,
            controller=controller,
        )
        runtime = policy.inference_module
        policy.total_inference_cnt = 0
        policy.state_processor._prepare_low_state()
        policy.reset()
        policy.state_dict.update(control_mode="policy", paused=False)
        peak_action = 0.0
        output_shapes: dict[str, list[int]] = {}
        for frame in range(steps):
            robot_io.frame = frame
            policy.total_inference_cnt = frame
            if not policy.state_processor._prepare_low_state():
                raise RuntimeError("Synthetic robot state is missing")
            policy.update()
            observations, _ = policy.prepare_obs_for_rl()
            _check_input_contract(runtime, observations)
            # Inspect every graph output (including MimicLite's auxiliary head).
            raw_outputs = runtime(observations)
            for key, output in raw_outputs.items():
                array = np.asarray(output)
                if not np.isfinite(array).all():
                    raise ValueError(f"ONNX output {key} contains non-finite values at step {frame}")
                output_shapes[str(key)] = list(array.shape)
            policy.state_dict.update(observations)
            policy.state_dict["is_init"] = np.zeros(1, dtype=bool)
            action, q_target, policy.state_dict = policy.policy(policy.state_dict)
            if np.asarray(action).shape != (29,) or not np.isfinite(action).all():
                raise ValueError("Runtime action must contain 29 finite values")
            expected_target = values["default_joint_pos"].copy()
            for index, name in enumerate(config["policy_joint_names"]):
                expected_target[canonical_names.index(name)] += action[index] * values["action_scale"][index]
            np.testing.assert_allclose(q_target, expected_target, rtol=1e-6, atol=1e-6)
            policy.state_dict.update(action=action, q_target=q_target, control_mode="policy")
            policy.action_manager.send_command(q_target, np.zeros(29), np.zeros(29))
            command = robot_io.commands[-1]
            np.testing.assert_allclose(command["kp"], values["joint_kp"])
            np.testing.assert_allclose(command["kd"], values["joint_kd"])
            peak_action = max(peak_action, float(np.max(np.abs(action))))
        return {
            "config": str(config_path),
            "model_path": str(model_path.resolve()),
            "backend": "onnx-cpu",
            "input_shapes": {
                str(key): list(shape) for key, shape in zip(runtime.in_keys, runtime.input_shapes)
            },
            "onnx_input_names": [item.name for item in runtime.ort_session.get_inputs()],
            "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
            "output_shapes": output_shapes,
            "action_shape": list(np.asarray(action).shape),
            "steps": steps,
            "finite": True,
            "joint_mapping": "29 G1 joints checked in canonical motor order",
            "observation_joint_order": list(config["joint_names_simulation"]),
            "policy_joint_order": list(config["policy_joint_names"]),
            "implicit_zero_default_joints": implicit_zero_joints,
            "peak_abs_action": peak_action,
            "hardware_commands_sent": 0,
            "validation_scope": "synthetic observations and inference; not a stability or hardware test",
        }
    finally:
        controller.close()
        robot_io.close()
