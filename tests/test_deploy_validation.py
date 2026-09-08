from __future__ import annotations

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy import deploy_validation as validation


@pytest.fixture
def config():
    robot = get_robot_cfg("g1")
    names = list(robot.joint_names)
    return {
        "model_path": "policy.onnx",
        "joint_names_simulation": names,
        # Reversal makes the action-to-motor mapping nontrivial.
        "policy_joint_names": list(reversed(names)),
        "body_names_simulation": list(robot.body_names),
        "default_joint_pos": {name: index * 0.001 for index, name in enumerate(names)},
        "joint_kp": {name: 10 + index for index, name in enumerate(names)},
        "joint_kd": {".*": 1.0},
        "action_scale": [0.1 + index * 0.001 for index in range(29)],
        "observation": {
            "policy": {"joint_pos": {"_target_": "mimic_lite.joint_pos_history", "history_steps": [0]}}
        },
        "motion": {"future_steps": [0, 1], "motion_backend": "zmq"},
    }


def _save_policy(tmp_path, config, *, input_width=29, auxiliary_nan=False):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))
    inputs = [helper.make_tensor_value_info("policy", TensorProto.FLOAT, [input_width])]
    outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])]
    actions = numpy_helper.from_array(np.arange(29, dtype=np.float32).reshape(1, 29) * 0.01)
    nodes = [helper.make_node("Constant", [], ["action"], value=actions)]
    if auxiliary_nan:
        outputs.append(helper.make_tensor_value_info("auxiliary", TensorProto.FLOAT, [1]))
        nodes.append(helper.make_node(
            "Constant", [], ["auxiliary"],
            value=numpy_helper.from_array(np.asarray([np.nan], dtype=np.float32)),
        ))
    graph = helper.make_graph(nodes, "offline-validation-test", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "policy.onnx")
    return path


@pytest.fixture
def no_live_transports(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline validation attempted to construct a live transport")

    monkeypatch.setattr("sim2real.rl_policy.base_policy.create_robot_io", forbidden)
    monkeypatch.setattr("sim2real.rl_policy.base_policy.BasePolicy._build_controller", forbidden)
    monkeypatch.setattr("sim2real.rl_policy.tracking.RealtimeMotionBuffer", forbidden)
    monkeypatch.setattr("sim2real.rl_policy.tracking.RealtimeSmplMotionBuffer", forbidden)
    monkeypatch.setenv("SIM2REAL_ORT_NUM_THREADS", "1")

    def fake_fk(robot_cfg, *, joint_names, body_names):
        quats = np.zeros((len(body_names), 4), dtype=np.float32)
        quats[:, 0] = 1
        return (
            np.asarray(robot_cfg.default_qpos[7:], dtype=np.float32),
            np.zeros((len(body_names), 3), dtype=np.float32), quats,
        )

    monkeypatch.setattr(validation, "_default_frame_from_robot_cfg", fake_fk)


@pytest.mark.parametrize("model_path", ["policy.onnx", None])
def test_preflight_runs_real_onnx_and_motor_mapping_without_transports(tmp_path, config, no_live_transports, model_path):
    config["model_path"] = model_path
    path = _save_policy(tmp_path, config)
    report = validation.check_policy(path, steps=3)
    assert report["finite"] is True
    assert report["action_shape"] == [29]
    assert report["input_shapes"] == {"policy": [29]}
    assert report["steps"] == 3
    assert report["hardware_commands_sent"] == 0
    assert report["peak_abs_action"] == pytest.approx(0.28)
    json.dumps(report)


def test_preflight_rejects_input_shape_before_inference(tmp_path, config, no_live_transports):
    path = _save_policy(tmp_path, config, input_width=30)
    with pytest.raises(ValueError, match="does not match ONNX"):
        validation.check_policy(path)


def test_preflight_rejects_nan_in_auxiliary_output(tmp_path, config, no_live_transports):
    path = _save_policy(tmp_path, config, auxiliary_nan=True)
    with pytest.raises(ValueError, match="auxiliary contains non-finite"):
        validation.check_policy(path)


def test_preflight_rejects_metadata_that_renames_a_graph_input(tmp_path, config, no_live_transports):
    path = _save_policy(tmp_path, config)
    model = onnx.load(path.with_suffix(".onnx"))
    model.graph.input[0].name = "wrong_input"
    onnx.save(model, path.with_suffix(".onnx"))
    path.with_suffix(".json").write_text(json.dumps({"in_keys": ["policy"], "out_keys": ["action"]}))
    with pytest.raises(ValueError, match="disagrees with graph binding"):
        validation.check_policy(path)


@pytest.mark.parametrize("key", ["joint_kp", "joint_kd", "action_scale"])
def test_preflight_requires_complete_joint_parameter_coverage(config, key):
    config[key] = {config["policy_joint_names"][0]: 1.0}
    with pytest.raises(ValueError, match=f"{key} is missing joints"):
        validation._validate_config(config)


def test_preflight_accepts_mimiclite_sparse_defaults_as_zeros(tmp_path, config, no_live_transports):
    name = config["joint_names_simulation"][0]
    config["default_joint_pos"] = {name: -0.28}
    path = _save_policy(tmp_path, config)
    report = validation.check_policy(path, steps=2)
    assert len(report["implicit_zero_default_joints"]) == 28
    assert name not in report["implicit_zero_default_joints"]
    defaults = validation._validate_config(config)["default_joint_pos"]
    np.testing.assert_allclose(defaults, [-0.28] + [0.0] * 28)


def test_preflight_rejects_unknown_sparse_default_key(config):
    config["default_joint_pos"] = {"misspelled_joint": 0.0}
    with pytest.raises(ValueError, match="regular expressions"):
        validation._validate_config(config)


@pytest.mark.parametrize("key,value", [
    ("default_joint_pos", float("nan")),
    ("joint_kp", float("inf")),
    ("joint_kd", -1.0),
    ("action_scale", 0.0),
])
def test_preflight_rejects_invalid_motor_parameters(config, key, value):
    config[key] = {".*": value}
    with pytest.raises(ValueError, match=key):
        validation._validate_config(config)


def test_preflight_rejects_duplicate_policy_joints(config):
    config["policy_joint_names"][0] = config["policy_joint_names"][1]
    with pytest.raises(ValueError, match="each of the 29"):
        validation._validate_config(config)


def test_preflight_preserves_valid_simulation_observation_permutation(config):
    config["joint_names_simulation"] = list(reversed(config["joint_names_simulation"]))
    validation._validate_config(config)


def test_preflight_rejects_duplicate_simulation_joints(config):
    config["joint_names_simulation"][0] = config["joint_names_simulation"][1]
    with pytest.raises(ValueError, match="joint_names_simulation"):
        validation._validate_config(config)


def test_preflight_rejects_missing_model_without_loading_runtime(tmp_path, config, monkeypatch):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(validation, "_SyntheticTracking", lambda *args, **kwargs: pytest.fail("constructed runtime"))
    with pytest.raises(FileNotFoundError, match="Policy ONNX is missing"):
        validation.check_policy(path)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_preflight_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="positive integer"):
        validation.check_policy("unused.yaml", steps=steps)
