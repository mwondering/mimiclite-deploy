from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.math import matrix_from_quat, quat_rotate_inverse_numpy, quat_rotate_numpy


SPV5_2_HISTORY_LENGTH = 50
SPV5_2_REFERENCE_STEPS = tuple(range(-42, 8))
SPV5_2_REFERENCE_FRAME_DIM = 3 + 6 + 29
SPV5_2_REFERENCE_INPUT_DIM = len(SPV5_2_REFERENCE_STEPS) * SPV5_2_REFERENCE_FRAME_DIM
SPV5_2_KEY_BODY_COUNT = 13
SPV5_2_KEY_BODY_DIM = SPV5_2_KEY_BODY_COUNT * (3 + 6 + 3 + 3)
SPV5_2_ESTIMATOR_HISTORY_DIM = SPV5_2_HISTORY_LENGTH * (29 + 29 + 3 + 3 + 29 + 29)

SPV5_2_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_DEFAULT_KINEMATICS_PATH = (
    Path(__file__).resolve().parent / "assets" / "g1_sp_tracking_kinematics.xml"
)
_GRAVITY_W = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass(frozen=True)
class _KeypointSpec:
    name: str
    body_name: str
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    correction_body_name: str | None = None
    correction_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)


_DEFAULT_KEYPOINT_SPECS = (
    _KeypointSpec("left_hip", "left_hip_yaw_link"),
    _KeypointSpec("left_knee", "left_knee_link"),
    _KeypointSpec("left_foot", "left_ankle_roll_link"),
    _KeypointSpec("right_hip", "right_hip_yaw_link"),
    _KeypointSpec("right_knee", "right_knee_link"),
    _KeypointSpec("right_foot", "right_ankle_roll_link"),
    _KeypointSpec("head", "torso_link", local_pos=(0.01, 0.0, 0.41)),
    _KeypointSpec("left_shoulder", "left_shoulder_yaw_link"),
    _KeypointSpec("left_wrist", "left_wrist_roll_link"),
    _KeypointSpec(
        "left_hand",
        "left_wrist_yaw_link",
        local_pos=(0.116, 0.0, 0.0),
        correction_body_name="left_wrist_pitch_link",
        correction_local_pos=(0.005, 0.0, 0.0),
    ),
    _KeypointSpec("right_shoulder", "right_shoulder_yaw_link"),
    _KeypointSpec("right_wrist", "right_wrist_roll_link"),
    _KeypointSpec(
        "right_hand",
        "right_wrist_yaw_link",
        local_pos=(0.116, 0.0, 0.0),
        correction_body_name="right_wrist_pitch_link",
        correction_local_pos=(0.005, 0.0, 0.0),
    ),
)


def _resolve_asset_path(env: Any, value: str | Path | None) -> Path:
    if value is None:
        return _DEFAULT_KINEMATICS_PATH
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    policy_config = getattr(getattr(env, "args", None), "policy_config", None)
    if not path.is_absolute() and policy_config is not None:
        candidates.append(Path(policy_config).expanduser().resolve().parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"SPV5-2 kinematics asset {value!r} was not found; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    identity = np.zeros_like(value)
    identity[..., 0] = 1.0
    return np.where(norm > 1.0e-8, value / np.maximum(norm, 1.0e-8), identity)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    matrix = matrix_from_quat(_normalize_quat(quat))
    return matrix[..., :, :2].swapaxes(-2, -1).reshape(*quat.shape[:-1], 6).astype(
        np.float32,
        copy=False,
    )


def _parse_keypoint_specs(
    raw_specs: Sequence[Mapping[str, Any]] | None,
) -> tuple[_KeypointSpec, ...]:
    if raw_specs is None:
        return _DEFAULT_KEYPOINT_SPECS
    specs: list[_KeypointSpec] = []
    for raw in raw_specs:
        data = dict(raw)
        body_name = str(data.get("body_name", data.get("asset_body_name", "")))
        correction_name = data.get(
            "correction_body_name", data.get("asset_correction_body_name")
        )
        specs.append(
            _KeypointSpec(
                name=str(data["name"]),
                body_name=body_name,
                local_pos=tuple(float(v) for v in data.get("local_pos", (0.0, 0.0, 0.0))),
                local_quat=tuple(
                    float(v) for v in data.get("local_quat", (1.0, 0.0, 0.0, 0.0))
                ),
                correction_body_name=(str(correction_name) if correction_name else None),
                correction_local_pos=tuple(
                    float(v)
                    for v in data.get("correction_local_pos", (0.0, 0.0, 0.0))
                ),
            )
        )
    if len(specs) != SPV5_2_KEY_BODY_COUNT:
        raise ValueError(
            f"SPV5-2 expects {SPV5_2_KEY_BODY_COUNT} semantic keypoints, got {len(specs)}"
        )
    return tuple(specs)


class _SPV52RobotKinematics:
    """MuJoCo FK/Jacobian implementation of the training-side analytic FK contract."""

    def __init__(
        self,
        xml_path: Path,
        joint_names: Sequence[str],
        keypoint_specs: tuple[_KeypointSpec, ...],
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(str(name) for name in joint_names)
        self.keypoint_specs = keypoint_specs

        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        missing_joints = [
            name for name, joint_id in zip(self.joint_names, joint_ids, strict=True) if joint_id < 0
        ]
        if missing_joints:
            raise ValueError(f"SPV5-2 kinematics asset is missing joints: {missing_joints}")
        self.qpos_addresses = np.asarray(
            [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32
        )
        self.qvel_addresses = np.asarray(
            [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32
        )

        physical_names: list[str] = []
        for spec in keypoint_specs:
            for name in (spec.body_name, spec.correction_body_name):
                if name is not None and name not in physical_names:
                    physical_names.append(name)
        self.physical_names = tuple(physical_names)
        body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.physical_names
        ]
        missing_bodies = [
            name for name, body_id in zip(self.physical_names, body_ids, strict=True) if body_id < 0
        ]
        if missing_bodies:
            raise ValueError(f"SPV5-2 kinematics asset is missing bodies: {missing_bodies}")
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        body_index = {name: index for index, name in enumerate(self.physical_names)}
        self.parent_indices = np.asarray(
            [body_index[spec.body_name] for spec in keypoint_specs], dtype=np.int32
        )
        self.correction_indices = np.asarray(
            [body_index[spec.correction_body_name or spec.body_name] for spec in keypoint_specs],
            dtype=np.int32,
        )
        self.local_pos = np.asarray([spec.local_pos for spec in keypoint_specs], dtype=np.float32)
        self.local_quat = np.asarray(
            [spec.local_quat for spec in keypoint_specs], dtype=np.float32
        )
        self.correction_local_pos = np.asarray(
            [spec.correction_local_pos for spec in keypoint_specs], dtype=np.float32
        )
        self.jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)

    def compute(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        root_ang_vel_b: np.ndarray,
    ) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(len(self.joint_names))
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(len(self.joint_names))
        root_ang_vel_b = np.asarray(root_ang_vel_b, dtype=np.float32).reshape(3)

        self.data.qpos.fill(0.0)
        self.data.qvel.fill(0.0)
        self.data.qpos[3] = 1.0
        self.data.qpos[self.qpos_addresses] = joint_pos
        self.data.qvel[3:6] = root_ang_vel_b
        self.data.qvel[self.qvel_addresses] = joint_vel
        mujoco.mj_forward(self.model, self.data)

        body_pos = np.asarray(self.data.xpos[self.body_ids], dtype=np.float32).copy()
        body_quat = np.asarray(self.data.xquat[self.body_ids], dtype=np.float32).copy()
        body_lin_vel = np.empty((len(self.body_ids), 3), dtype=np.float32)
        body_ang_vel = np.empty_like(body_lin_vel)
        for index, body_id in enumerate(self.body_ids):
            mujoco.mj_jacBody(
                self.model,
                self.data,
                self.jac_pos,
                self.jac_rot,
                int(body_id),
            )
            body_lin_vel[index] = self.jac_pos @ self.data.qvel
            body_ang_vel[index] = self.jac_rot @ self.data.qvel

        parent_pos = body_pos[self.parent_indices]
        parent_quat = body_quat[self.parent_indices]
        parent_lin_vel = body_lin_vel[self.parent_indices]
        parent_ang_vel = body_ang_vel[self.parent_indices]
        correction_quat = body_quat[self.correction_indices]
        correction_ang_vel = body_ang_vel[self.correction_indices]

        offset = quat_rotate_numpy(parent_quat, self.local_pos)
        correction = quat_rotate_numpy(correction_quat, self.correction_local_pos)
        key_pos = parent_pos + offset + correction
        key_quat = _normalize_quat(_quat_mul(parent_quat, self.local_quat))
        key_lin_vel = (
            parent_lin_vel
            + np.cross(parent_ang_vel, offset)
            + np.cross(correction_ang_vel, correction)
        )
        key_ang_vel = parent_ang_vel - root_ang_vel_b.reshape(1, 3)
        packed = np.concatenate(
            (
                key_pos.reshape(-1),
                _quat_to_rot6d(key_quat).reshape(-1),
                key_lin_vel.reshape(-1),
                key_ang_vel.reshape(-1),
            )
        ).astype(np.float32, copy=False)
        if packed.shape != (SPV5_2_KEY_BODY_DIM,):
            raise RuntimeError(
                f"SPV5-2 key-body observation has shape {packed.shape}, "
                f"expected {(SPV5_2_KEY_BODY_DIM,)}"
            )
        return packed.reshape(1, -1)


class _SPV52ObservationCore:
    def __init__(
        self,
        env: Any,
        *,
        kinematics_path: str | Path | None,
        joint_names: Sequence[str],
        history_length: int,
        reference_steps: Sequence[int],
        root_body_name: str,
        keypoint_specs: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        self.env = env
        self.joint_names = tuple(str(name) for name in joint_names)
        self.history_length = int(history_length)
        self.reference_steps = tuple(int(step) for step in reference_steps)
        self.root_body_name = str(root_body_name)
        self.keypoint_specs = _parse_keypoint_specs(keypoint_specs)
        if self.joint_names != SPV5_2_JOINT_NAMES:
            raise ValueError(
                "SPV5-2 requires the MuJoCo G1 joint order; got "
                f"{self.joint_names}"
            )
        if self.history_length != SPV5_2_HISTORY_LENGTH:
            raise ValueError(
                f"SPV5-2 history length must be {SPV5_2_HISTORY_LENGTH}, "
                f"got {self.history_length}"
            )
        if self.reference_steps != SPV5_2_REFERENCE_STEPS:
            raise ValueError(
                f"SPV5-2 reference steps must be {SPV5_2_REFERENCE_STEPS}, "
                f"got {self.reference_steps}"
            )

        resolved_path = _resolve_asset_path(env, kinematics_path)
        self.kinematics = _SPV52RobotKinematics(
            resolved_path,
            self.joint_names,
            self.keypoint_specs,
        )
        self._layout: tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]] | None = None
        self._state_joint_indices: np.ndarray | None = None
        self._motion_joint_indices: np.ndarray | None = None
        self._motion_step_indices: np.ndarray | None = None
        self._root_body_index: int | None = None
        self._default_joint_pos = np.zeros(len(self.joint_names), dtype=np.float32)

        term_dims = (29, 29, 3, 3, 29, 29)
        self._history = [
            np.zeros((self.history_length, dim), dtype=np.float32) for dim in term_dims
        ]
        self._history_initialized = False
        self._last_update_token: tuple[int, int, int] | None = None
        self.robot_root_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.estimator_history = np.zeros((1, SPV5_2_ESTIMATOR_HISTORY_DIM), dtype=np.float32)
        self.reference_encoder_input = np.zeros(
            (1, SPV5_2_REFERENCE_INPUT_DIM), dtype=np.float32
        )
        self.robot_key_body = np.zeros((1, SPV5_2_KEY_BODY_DIM), dtype=np.float32)
        self._refresh_layout()

    def reset(self) -> None:
        for history in self._history:
            history.fill(0.0)
        self._history_initialized = False
        self._last_update_token = None
        self.robot_root_quat[:] = (1.0, 0.0, 0.0, 0.0)
        self.estimator_history.fill(0.0)
        self.reference_encoder_input.fill(0.0)
        self.robot_key_body.fill(0.0)

    def _refresh_layout(self) -> None:
        state_names = tuple(str(name) for name in self.env.state_processor.joint_names)
        motion_names = tuple(str(name) for name in getattr(self.env, "motion_joint_names", ()))
        motion_steps = tuple(
            int(step) for step in np.asarray(getattr(self.env, "motion_future_steps", ())).reshape(-1)
        )
        layout = (state_names, motion_names, motion_steps)
        if layout == self._layout:
            return

        policy_names = tuple(str(name) for name in self.env.policy_joint_names)
        if policy_names != self.joint_names:
            raise ValueError(
                "SPV5-2 policy action order must match the training MuJoCo order; "
                f"expected={self.joint_names}, got={policy_names}"
            )
        missing_state = [name for name in self.joint_names if name not in state_names]
        missing_motion = [name for name in self.joint_names if name not in motion_names]
        missing_steps = [step for step in self.reference_steps if step not in motion_steps]
        body_names = tuple(str(name) for name in getattr(self.env, "motion_body_names", ()))
        if missing_state or missing_motion or missing_steps or self.root_body_name not in body_names:
            raise ValueError(
                "SPV5-2 runtime layout mismatch: "
                f"missing_state_joints={missing_state}, missing_motion_joints={missing_motion}, "
                f"missing_reference_steps={missing_steps}, root_body={self.root_body_name!r}, "
                f"available_bodies={body_names}"
            )

        self._state_joint_indices = np.asarray(
            [state_names.index(name) for name in self.joint_names], dtype=np.int32
        )
        self._motion_joint_indices = np.asarray(
            [motion_names.index(name) for name in self.joint_names], dtype=np.int32
        )
        self._motion_step_indices = np.asarray(
            [motion_steps.index(step) for step in self.reference_steps], dtype=np.int32
        )
        self._root_body_index = body_names.index(self.root_body_name)

        simulation_names = tuple(str(name) for name in self.env.joint_names_simulation)
        defaults = np.asarray(self.env.default_dof_angles, dtype=np.float32).reshape(-1)
        if defaults.size != len(simulation_names):
            raise ValueError(
                "SPV5-2 default joint position layout mismatch: "
                f"{defaults.size} values for {len(simulation_names)} names"
            )
        default_by_name = dict(zip(simulation_names, defaults, strict=True))
        missing_defaults = [name for name in self.joint_names if name not in default_by_name]
        if missing_defaults:
            raise ValueError(f"SPV5-2 default pose is missing joints: {missing_defaults}")
        self._default_joint_pos = np.asarray(
            [default_by_name[name] for name in self.joint_names], dtype=np.float32
        )
        self._layout = layout

    def _update_token(self) -> tuple[int, int, int]:
        state = self.env.state_processor
        tick = int(getattr(state, "low_state_tick", -1))
        latest_state = getattr(state, "latest_state", None)
        if tick < 0 and latest_state is not None:
            tick = int(getattr(latest_state, "tick", -1))
        motion_t = np.asarray(getattr(self.env, "motion_t", (-1,))).reshape(-1)
        motion_index = int(motion_t[0]) if motion_t.size else -1
        return int(getattr(self.env, "total_inference_cnt", -1)), tick, motion_index

    def _current_torque(self) -> np.ndarray:
        state = self.env.state_processor
        torque = getattr(state, "joint_torque", None)
        if torque is None:
            latest_state = getattr(state, "latest_state", None)
            torque = getattr(latest_state, "joint_torque", None)
        if torque is None:
            torque = np.zeros(len(state.joint_names), dtype=np.float32)
        value = np.asarray(torque, dtype=np.float32).reshape(-1)
        if value.size != len(state.joint_names):
            raise ValueError(
                f"SPV5-2 expected {len(state.joint_names)} joint torques, got {value.size}"
            )
        assert self._state_joint_indices is not None
        return value[self._state_joint_indices]

    def _append_history(self, samples: tuple[np.ndarray, ...]) -> None:
        if not self._history_initialized:
            for history, sample in zip(self._history, samples, strict=True):
                history[:] = sample.reshape(1, -1)
            self._history_initialized = True
        else:
            for history, sample in zip(self._history, samples, strict=True):
                history[:-1] = history[1:]
                history[-1] = sample
        self.estimator_history[0] = np.concatenate(
            [history.reshape(-1) for history in self._history]
        )

    def _reference_input(self) -> np.ndarray:
        motion_data = getattr(self.env, "motion_data", None)
        if motion_data is None:
            raise ValueError("SPV5-2 reference observation requires motion_data")
        assert self._motion_step_indices is not None
        assert self._motion_joint_indices is not None
        assert self._root_body_index is not None
        step_ids = self._motion_step_indices
        root_pos = np.asarray(
            motion_data.body_pos_w[:, step_ids, self._root_body_index], dtype=np.float32
        )
        root_quat = _normalize_quat(
            np.asarray(
                motion_data.body_quat_w[:, step_ids, self._root_body_index],
                dtype=np.float32,
            )
        )
        joint_pos = np.asarray(
            motion_data.joint_pos[:, step_ids][:, :, self._motion_joint_indices],
            dtype=np.float32,
        )
        frames = np.concatenate((root_pos, _quat_to_rot6d(root_quat), joint_pos), axis=-1)
        value = frames.reshape(1, -1).astype(np.float32, copy=False)
        if value.shape != (1, SPV5_2_REFERENCE_INPUT_DIM):
            raise RuntimeError(
                f"SPV5-2 reference input has shape {value.shape}, "
                f"expected {(1, SPV5_2_REFERENCE_INPUT_DIM)}"
            )
        return value

    def update_once(self, data: dict[str, Any]) -> None:
        token = self._update_token()
        if token == self._last_update_token:
            return
        self._refresh_layout()
        assert self._state_joint_indices is not None
        state = self.env.state_processor
        joint_pos = np.asarray(state.joint_pos, dtype=np.float32).reshape(-1)[
            self._state_joint_indices
        ]
        joint_vel = np.asarray(state.joint_vel, dtype=np.float32).reshape(-1)[
            self._state_joint_indices
        ]
        root_quat = _normalize_quat(
            np.asarray(state.root_quat_w, dtype=np.float32).reshape(1, 4)
        )
        root_ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)
        gravity = quat_rotate_inverse_numpy(
            root_quat,
            np.broadcast_to(_GRAVITY_W, (1, 3)),
        )[0].astype(np.float32, copy=False)
        last_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if last_action.size != len(self.joint_names):
            raise ValueError(
                f"SPV5-2 expected {len(self.joint_names)} previous actions, "
                f"got {last_action.size}"
            )
        torque = self._current_torque()

        self._append_history(
            (
                joint_pos - self._default_joint_pos,
                joint_vel,
                gravity,
                root_ang_vel,
                last_action,
                torque,
            )
        )
        self.robot_root_quat[:] = root_quat
        self.reference_encoder_input[:] = self._reference_input()
        self.robot_key_body[:] = self.kinematics.compute(
            joint_pos,
            joint_vel,
            root_ang_vel,
        )
        for name in (
            "robot_root_quat",
            "estimator_history",
            "reference_encoder_input",
            "robot_key_body",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"SPV5-2 observation {name} contains NaN or Inf")
        self._last_update_token = token

    def output(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _core_key(
    *,
    kinematics_path: str | Path | None,
    joint_names: Sequence[str],
    history_length: int,
    reference_steps: Sequence[int],
    root_body_name: str,
    keypoint_specs: Sequence[Mapping[str, Any]] | None,
) -> tuple[Any, ...]:
    specs = _parse_keypoint_specs(keypoint_specs)
    return (
        "sp_tracking_spv5_2",
        str(kinematics_path) if kinematics_path is not None else None,
        tuple(str(name) for name in joint_names),
        int(history_length),
        tuple(int(step) for step in reference_steps),
        str(root_body_name),
        specs,
    )


def _get_core(env: Any, **kwargs: Any) -> _SPV52ObservationCore:
    key = _core_key(**kwargs)
    cache = getattr(env, "_sp_tracking_spv5_2_core_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_sp_tracking_spv5_2_core_cache", cache)
    if key not in cache:
        cache[key] = _SPV52ObservationCore(env, **kwargs)
    return cache[key]


class _SPV52Observation(Observation, namespace="sp_tracking"):
    output_name: str

    def __init__(
        self,
        kinematics_path: str | Path | None = None,
        joint_names: Sequence[str] = SPV5_2_JOINT_NAMES,
        history_length: int = SPV5_2_HISTORY_LENGTH,
        reference_steps: Sequence[int] = SPV5_2_REFERENCE_STEPS,
        root_body_name: str = "pelvis",
        keypoint_specs: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.core = _get_core(
            self.env,
            kinematics_path=kinematics_path,
            joint_names=joint_names,
            history_length=history_length,
            reference_steps=reference_steps,
            root_body_name=root_body_name,
            keypoint_specs=keypoint_specs,
        )

    def reset(self) -> None:
        self.core.reset()

    def update(self, data: dict[str, Any]) -> None:
        self.core.update_once(data)

    def compute(self) -> np.ndarray:
        return self.core.output(self.output_name)


class spv5_2_robot_root_quat(_SPV52Observation, namespace="sp_tracking"):
    output_name = "robot_root_quat"


class spv5_2_estimator_history(_SPV52Observation, namespace="sp_tracking"):
    output_name = "estimator_history"


class spv5_2_reference_encoder_input(_SPV52Observation, namespace="sp_tracking"):
    output_name = "reference_encoder_input"


class spv5_2_robot_key_body(_SPV52Observation, namespace="sp_tracking"):
    output_name = "robot_key_body"
