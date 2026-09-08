"""G1 hardware loop shared by the MimicLite, SP Tracking and HEFT profiles.

Use ``python -m scripts.g1.deploy`` to validate artifacts before opening DDS.
The policy's observation/action implementation remains the shared Tracking one.
"""
from __future__ import annotations

import time

import numpy as np
from loguru import logger

from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.tracking import Tracking


class DeferredRobotIO(RobotIO):
    """Build all model/observation resources before attaching a hardware backend."""

    def __init__(self) -> None:
        self.backend: RobotIO | None = None

    def read_state(self) -> RobotState | None:
        return None if self.backend is None else self.backend.read_state()

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        if self.backend is None:
            raise RuntimeError("RobotIO has not been attached")
        self.backend.write_command(q_target, dq_target, tau_ff, kp, kd)

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    @property
    def emergency_stop_reason(self):
        return getattr(self.backend, "emergency_stop_reason", None)

    def poll_stop(self):
        if self.backend is not None:
            poll = getattr(self.backend, "poll_stop", None)
            if poll is not None:
                poll()


class RealTracking(Tracking):
    def __init__(self, args, *, robot_io, controller, stream_timeout=1.0, startup_timeout=15.0):
        self.stream_timeout = float(stream_timeout)
        self.startup_timeout = float(startup_timeout)
        self.fault_reason: str | None = None
        self._hold_target: np.ndarray | None = None
        self._state_wait_started: float | None = None
        self._init_completed = False
        try:
            super().__init__(args, robot_io=robot_io, controller=controller)
        except BaseException:
            buffer = getattr(self, "motion_buffer", None)
            if buffer is not None:
                buffer.close()
            raise
        self.total_inference_cnt = 0

    def stream_ages(self) -> dict[str, float | None]:
        now = time.monotonic()
        receipts = {
            "motion": self.motion_buffer.last_receive_monotonic,
            "controller": self.controller.last_receive_monotonic,
        }
        return {name: None if stamp is None else max(0.0, now - stamp)
                for name, stamp in receipts.items()}

    def _require_streams(self) -> None:
        bad = {name: age for name, age in self.stream_ages().items()
               if age is None or age > self.stream_timeout}
        if bad:
            raise RuntimeError(f"PICO ZMQ stream missing/stale (seconds): {bad}")

    def wait_for_streams(self) -> None:
        """Drain startup button edges without requesting any robot control mode."""
        deadline = time.monotonic() + self.startup_timeout
        next_log = 0.0
        while True:
            self.controller.get_control_mode()
            ages = self.stream_ages()
            if all(age is not None and age <= self.stream_timeout for age in ages.values()):
                self._update_motion_data(paused=True)
                logger.info("PICO motion and controller streams received")
                return
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"PICO startup timed out before opening RobotIO: {ages}")
            if now >= next_log:
                logger.info("Waiting for PICO streams before opening RobotIO: {}", ages)
                next_log = now + 1.0
            time.sleep(0.02)

    def _update_motion_data(self, *, paused: bool = False) -> None:
        super()._update_motion_data(paused=paused)
        if len(set(self.motion_joint_names)) != len(self.motion_joint_names):
            raise ValueError("PICO reference contains duplicate joint names")
        missing = set(self.policy_joint_names) - set(self.motion_joint_names)
        if missing:
            raise ValueError(f"PICO reference is missing policy joints: {sorted(missing)}")
        root_name = self.motion_config.get("root_body_name", "pelvis")
        if root_name not in self.motion_body_names:
            raise ValueError(f"PICO reference is missing root body {root_name}")
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                     "body_lin_vel_w", "body_ang_vel_w"):
            if not np.all(np.isfinite(getattr(self.motion_data, name))):
                raise ValueError(f"Non-finite PICO reference: {name}")
        norms = np.linalg.norm(self.motion_data.body_quat_w, axis=-1)
        if np.any(np.abs(norms - 1.0) > 0.05):
            raise ValueError("PICO reference contains invalid body quaternions")

    def prepare_obs_for_rl(self):
        observations, components = super().prepare_obs_for_rl()
        for name, value in observations.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite observation: {name}")
        return observations, components

    def set_init_mode(self, *, source: str) -> None:
        if self.fault_reason is None:
            self._init_completed = False
            super().set_init_mode(source=source)

    def set_policy_mode(self, *, source: str) -> None:
        if self.fault_reason is not None:
            return
        if not self._init_completed:
            logger.warning("Press A and finish initialization before A+B")
            return
        # A restart must not feed the last run's action to a different history.
        self.state_dict["action"] = np.zeros(self.num_actions, dtype=np.float32)
        super().set_policy_mode(source=source)

    def _latch_fault(self, error: Exception) -> None:
        if self.fault_reason is None:
            self.fault_reason = str(error)
            logger.error("G1 policy fault: {}. Holding joints; restart required.", error)
            if self.state_processor.latest_state is not None:
                self._hold_target = self.state_processor.joint_pos.copy()
        self.state_dict["control_mode"] = "zero"
        self.state_dict["action"] = np.zeros(self.num_actions, dtype=np.float32)

    def _send_hold(self) -> None:
        if self._hold_target is not None:
            zeros = np.zeros(self.num_dofs, dtype=np.float32)
            self.action_manager.send_command(self._hold_target, zeros, zeros)

    def step(self) -> None:
        if getattr(self.robot_io, "emergency_stop_reason", None) is not None:
            # Before state reads, PICO handling, faults and inference.
            self.robot_io.poll_stop()
            return
        # A latched fault never resumes inference on a delayed A+B packet.
        if self.fault_reason is not None:
            self._send_hold()
            return
        try:
            if not self.state_processor._prepare_low_state():
                if self._state_wait_started is None:
                    self._state_wait_started = time.monotonic()
                if self.state_processor.latest_state is not None:
                    raise RuntimeError("G1 low state became unavailable")
                if time.monotonic() - self._state_wait_started > self.startup_timeout:
                    raise TimeoutError("No valid G1 low state received")
                return
            # Read valid state before controller-triggered history/reset updates.
            self.process_controllers()
            self._require_streams()
            self.update()
            obs_dict, obs_components = self.prepare_obs_for_rl()
            control_mode = self.state_dict["control_mode"]
            self.state_dict.update(obs_dict)
            self.state_dict["is_init"] = np.zeros(1, dtype=bool)
            action, q_target, self.state_dict = self.policy(self.state_dict)
            self.state_dict["action"] = action
            self.state_dict["q_target"] = q_target
            self.state_dict["control_mode"] = control_mode
            if control_mode == "init":
                q_target = self.get_init_target()
                self._init_completed = self.init_count > 500
            elif control_mode == "zero":
                q_target = self.state_processor.joint_pos.copy()
            elif control_mode != "policy":
                raise ValueError(f"Unknown control mode: {control_mode}")
            if q_target.shape != (self.num_dofs,) or not np.all(np.isfinite(q_target)):
                raise ValueError("Invalid G1 joint targets")
            zeros = np.zeros(self.num_dofs, dtype=np.float32)
            self.action_manager.send_command(q_target, zeros, zeros)
            self._append_record_frame(
                obs_dict=obs_dict, obs_components=obs_components, action=action,
                q_target=q_target, cmd_q=q_target, cmd_dq=zeros, cmd_tau=zeros,
            )
        except Exception as error:
            self._latch_fault(error)
            self._send_hold()
            if self._hold_target is None:
                # There has never been a usable state from which to hold.
                raise

    def run(self) -> None:
        logger.info("G1 ready: A initializes (~10 s), release then A+B runs policy; X controls reference")
        deadline = time.monotonic()
        try:
            while True:
                self.step()
                self.total_inference_cnt += 1
                deadline += self.rl_dt
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # Do not issue a burst of old 50 Hz commands after a slow step.
                    deadline = time.monotonic()
                    if self.total_inference_cnt % 100 == 0:
                        logger.warning("G1 loop exceeded its 20 ms budget by {:.1f} ms", -delay * 1000)
        except KeyboardInterrupt:
            logger.info("Stopping G1 tracking")
        finally:
            try:
                if self.state_processor.latest_state is not None:
                    if self._hold_target is None:
                        self._hold_target = self.state_processor.joint_pos.copy()
                    self._send_hold()
            finally:
                self._save_recording()

    def close(self) -> None:
        try:
            self.motion_buffer.close()
        finally:
            try:
                self.controller.close()
            finally:
                self.robot_io.close()
