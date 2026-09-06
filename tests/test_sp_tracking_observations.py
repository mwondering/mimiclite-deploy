from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.sp_tracking import (
    SPV5_2_ESTIMATOR_HISTORY_DIM,
    SPV5_2_HISTORY_LENGTH,
    SPV5_2_JOINT_NAMES,
    SPV5_2_KEY_BODY_DIM,
    SPV5_2_REFERENCE_FRAME_DIM,
    SPV5_2_REFERENCE_INPUT_DIM,
    SPV5_2_REFERENCE_STEPS,
    _SPV52ObservationCore,
)


_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "adapt_sp_tracking_spv5_2.py"
_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "sim2real_sp_tracking_adapter",
    _ADAPTER_PATH,
)
assert _ADAPTER_SPEC is not None and _ADAPTER_SPEC.loader is not None
adapter = importlib.util.module_from_spec(_ADAPTER_SPEC)
_ADAPTER_SPEC.loader.exec_module(adapter)


def _fake_env() -> SimpleNamespace:
    num_steps = len(SPV5_2_REFERENCE_STEPS)
    num_joints = len(SPV5_2_JOINT_NAMES)
    root_pos = np.zeros((1, num_steps, 1, 3), dtype=np.float32)
    root_pos[0, :, 0, 0] = np.arange(num_steps, dtype=np.float32)
    root_pos[0, :, 0, 1] = -np.arange(num_steps, dtype=np.float32)
    root_quat = np.zeros((1, num_steps, 1, 4), dtype=np.float32)
    root_quat[..., 0] = 1.0
    joint_pos = np.arange(num_steps * num_joints, dtype=np.float32).reshape(
        1,
        num_steps,
        num_joints,
    )
    default = np.asarray(adapter.DEFAULT_JOINT_POS, dtype=np.float32)
    state = SimpleNamespace(
        joint_names=list(SPV5_2_JOINT_NAMES),
        joint_pos=default + np.arange(num_joints, dtype=np.float32) * 0.01,
        joint_vel=np.arange(num_joints, dtype=np.float32) * -0.02,
        joint_torque=np.arange(num_joints, dtype=np.float32) * 0.03,
        root_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        root_ang_vel_b=np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        low_state_tick=7,
        latest_state=None,
    )
    return SimpleNamespace(
        state_processor=state,
        policy_joint_names=list(SPV5_2_JOINT_NAMES),
        joint_names_simulation=list(SPV5_2_JOINT_NAMES),
        body_names_simulation=["pelvis"],
        default_dof_angles=default,
        motion_joint_names=list(SPV5_2_JOINT_NAMES),
        motion_body_names=["pelvis"],
        motion_future_steps=np.asarray(SPV5_2_REFERENCE_STEPS, dtype=np.int64),
        motion_data=SimpleNamespace(
            body_pos_w=root_pos,
            body_quat_w=root_quat,
            joint_pos=joint_pos,
        ),
        motion_t=np.asarray([0], dtype=np.int64),
        total_inference_cnt=0,
        args=SimpleNamespace(policy_config="/tmp/policy.yaml"),
    )


def _core(env: SimpleNamespace) -> _SPV52ObservationCore:
    return _SPV52ObservationCore(
        env,
        kinematics_path=None,
        joint_names=SPV5_2_JOINT_NAMES,
        history_length=SPV5_2_HISTORY_LENGTH,
        reference_steps=SPV5_2_REFERENCE_STEPS,
        root_body_name="pelvis",
        keypoint_specs=None,
    )


def test_spv5_2_contract_dimensions_and_registry() -> None:
    assert SPV5_2_ESTIMATOR_HISTORY_DIM == 6100
    assert SPV5_2_REFERENCE_INPUT_DIM == 1900
    assert SPV5_2_KEY_BODY_DIM == 195
    assert 4 + 6100 + 1900 + 195 == 8199
    for name, _dim in adapter.SEMANTIC_INPUTS:
        target = f"sp_tracking.spv5_2_{name}"
        assert Observation.resolve(target).__name__ == f"spv5_2_{name}"


def test_spv5_2_first_sample_backfills_term_major_history() -> None:
    env = _fake_env()
    core = _core(env)
    action = np.arange(29, dtype=np.float32) * 0.04
    core.update_once({"action": action})

    q_rel = env.state_processor.joint_pos - env.default_dof_angles
    terms = (
        q_rel,
        env.state_processor.joint_vel,
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        env.state_processor.root_ang_vel_b,
        action,
        env.state_processor.joint_torque,
    )
    expected = np.concatenate(
        [np.broadcast_to(term, (SPV5_2_HISTORY_LENGTH, term.size)).reshape(-1) for term in terms]
    )
    np.testing.assert_allclose(core.estimator_history[0], expected, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(
        core.robot_root_quat,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_spv5_2_core_updates_only_once_per_runtime_step_and_rolls_oldest_first() -> None:
    env = _fake_env()
    core = _core(env)
    core.update_once({"action": np.zeros(29, dtype=np.float32)})
    original = core.estimator_history.copy()

    env.state_processor.joint_pos += 1.0
    core.update_once({"action": np.ones(29, dtype=np.float32)})
    np.testing.assert_array_equal(core.estimator_history, original)

    env.total_inference_cnt += 1
    core.update_once({"action": np.ones(29, dtype=np.float32)})
    joint_history = core.estimator_history[0, : SPV5_2_HISTORY_LENGTH * 29].reshape(
        SPV5_2_HISTORY_LENGTH,
        29,
    )
    np.testing.assert_allclose(
        joint_history[:-1],
        original[0, : SPV5_2_HISTORY_LENGTH * 29].reshape(SPV5_2_HISTORY_LENGTH, 29)[1:],
    )
    np.testing.assert_allclose(
        joint_history[-1],
        env.state_processor.joint_pos - env.default_dof_angles,
    )


def test_spv5_2_reference_input_is_frame_major_with_column_major_rot6d() -> None:
    env = _fake_env()
    core = _core(env)
    core.update_once({"action": np.zeros(29, dtype=np.float32)})

    frames = core.reference_encoder_input.reshape(
        len(SPV5_2_REFERENCE_STEPS),
        SPV5_2_REFERENCE_FRAME_DIM,
    )
    np.testing.assert_array_equal(frames[:, :3], env.motion_data.body_pos_w[0, :, 0])
    np.testing.assert_array_equal(
        frames[:, 3:9],
        np.broadcast_to(
            np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            (len(SPV5_2_REFERENCE_STEPS), 6),
        ),
    )
    np.testing.assert_array_equal(frames[:, 9:], env.motion_data.joint_pos[0])


def test_spv5_2_shared_observations_reuse_one_stateful_core() -> None:
    env = _fake_env()
    observations = [
        Observation.resolve(f"sp_tracking.spv5_2_{name}")(env=env)
        for name, _dim in adapter.SEMANTIC_INPUTS
    ]
    assert len({id(observation.core) for observation in observations}) == 1

    for observation in observations:
        observation.update({"action": np.zeros(29, dtype=np.float32)})
    outputs = [observation.compute() for observation in observations]
    assert [output.shape for output in outputs] == [(1, dim) for _, dim in adapter.SEMANTIC_INPUTS]
    assert all(np.all(np.isfinite(output)) for output in outputs)


def test_spv5_2_generated_policy_yaml_preserves_source_action_contract() -> None:
    config = adapter._policy_yaml()
    names = list(SPV5_2_JOINT_NAMES)
    assert config["policy_joint_names"] == names
    assert config["motion"]["future_steps"] == list(SPV5_2_REFERENCE_STEPS)
    np.testing.assert_allclose(
        [config["default_joint_pos"][name] for name in names],
        adapter.DEFAULT_JOINT_POS,
    )
    np.testing.assert_allclose(
        [config["joint_kp"][name] for name in names],
        adapter.JOINT_KP,
    )
    np.testing.assert_allclose(
        [config["joint_kd"][name] for name in names],
        adapter.JOINT_KD,
    )
    np.testing.assert_allclose(
        [config["action_scale"][name] for name in names],
        adapter.ACTION_SCALE,
    )


@pytest.mark.parametrize("runtime_kind", ["sim2sim", "sim2real"])
@pytest.mark.parametrize("clip_actions", [None, 10.0])
def test_runtime_clips_action_before_scaling_and_history(
    monkeypatch: pytest.MonkeyPatch, runtime_kind: str, clip_actions: float | None
) -> None:
    if runtime_kind == "sim2real":
        import sim2real.rl_policy.base_policy as module

        runtime_class = module.BasePolicy
    else:
        import sim2real.sim_env.integrated_sim2sim as module

        runtime_class = module.IntegratedPolicyRuntime

    raw = np.linspace(-20.0, 20.0, 29, dtype=np.float32)
    monkeypatch.setattr(
        module, "build_inference_module",
        lambda *_args: lambda _inputs: {"action": raw.copy()},
    )
    runtime = object.__new__(runtime_class)
    runtime.policy_config = {} if clip_actions is None else {"clip_actions": clip_actions}
    runtime.inference_backend = "onnx-cpu"
    runtime.default_dof_angles = np.ones(29, dtype=np.float32)
    runtime.controlled_joint_indices = np.arange(29)[::-1]
    runtime.action_scale = np.full(29, 0.25, dtype=np.float32)
    runtime.setup_policy("unused.onnx")
    action, target, _ = runtime.policy({})
    expected = raw if clip_actions is None else np.clip(raw, -10.0, 10.0)
    # Returned actions feed the next observation; targets must use the same values.
    np.testing.assert_array_equal(action, expected)
    np.testing.assert_array_equal(target[::-1], 1.0 + 0.25 * expected)


def test_spv5_2a_resolved_training_config_metadata(tmp_path: Path) -> None:
    history_names = (
        "joint_pos",
        "joint_vel",
        "projected_gravity",
        "base_ang_vel",
        "last_action",
        "joint_torque",
    )
    config = {
        "agent": {"run_name": "SPV5-2A-test"},
        "checkpoint_path": "/logs/2026-08-31_SPV5-2A/checkpoint_26000.pt",
        "task": {
            "decimation": 4,
            "sim": {"timestep": 0.005},
            "robot": {
                "name": "g1",
                "anchor_body_name": "pelvis",
                "body_names": ["pelvis"],
            },
            "action": {"scale": "g1_action_scale", "use_default_offset": True},
            "variant": {"observation_profile": "spv5_2a_qpos_only_actor_fk"},
            "obs": {
                "observations": {
                    "estimator_history": {
                        "terms": [
                            {"name": name, "history_length": SPV5_2_HISTORY_LENGTH}
                            for name in history_names
                        ]
                    },
                    "robot_root_quat": {
                        "terms": [{"term": "spv5_robot_root_quat"}]
                    },
                    "reference_encoder_input": {
                        "terms": [{"term": "spv5_reference_encoder_input"}]
                    },
                    "robot_key_body": {
                        "terms": [{"term": "spv5_2_robot_key_body_state"}]
                    },
                }
            },
            "agent_overrides": {
                "obs_groups": {
                    "actor": [name for name, _dim in adapter.SEMANTIC_INPUTS]
                },
                "actor": {
                    "class_name": (
                        "sp_tracking.tasks.tracking.rl.spv5_2_models:"
                        "SPV52HeightContactEstimatorActor"
                    ),
                    "estimator_history_length": SPV5_2_HISTORY_LENGTH,
                    "reference_fps": 50.0,
                },
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    metadata = adapter._metadata_from_resolved_config(config_path, 36000)

    assert metadata["format"] == "motion_tracking_sim2real_resolved_config"
    assert metadata["run_name"] == "2026-08-31_SPV5-2A"
    assert metadata["iteration"] == 36000
    assert metadata["checkpoint"] == "checkpoint_36000.pt"
    assert metadata["policy_variant"] == "SPV52HeightContactEstimatorActor"
    assert metadata["observation_profile"] == "spv5_2a_qpos_only_actor_fk"
    assert metadata["control_dt_s"] == 0.02
    adapter._validate_source_metadata(metadata)
