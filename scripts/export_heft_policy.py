#!/usr/bin/env python3
"""Package the released motion_tracking G1 PMG policy for sim2real.

Run against a checkout/archive of motion_tracking's sim2real branch. The source
normalizer, student adaptor and actor remain inside one ONNX; only its external
input/output interface changes. No training code or PICO process is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from sim2real.config.robots import get_robot_cfg  # noqa: E402

SOURCE_DEPLOY_COMMIT = "0d5ba31e33397f3543d350d98b637e26d92f470a"
HISTORY_KEYS = (
    "root_angvel_history_steps",
    "projected_gravity_history_steps",
    "joint_pos_history_steps",
    "joint_vel_history_steps",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sizes(tracking: dict, joint_count: int) -> dict[str, int]:
    """Widths follow TrackingPolicyRaw._build_obs_modules(), in that order."""
    reference_count = len(tracking["future_steps"])
    return {
        "context": 1,
        "motion_command": (reference_count - 1) * 3 + reference_count * 6,
        "target_motion": reference_count * (2 * joint_count + 1 + 3),
        "proprioception": (
            len(tracking[HISTORY_KEYS[0]]) * 3
            + len(tracking[HISTORY_KEYS[1]]) * 3
            + len(tracking[HISTORY_KEYS[2]]) * joint_count
            + len(tracking[HISTORY_KEYS[3]]) * joint_count
            + int(tracking["prev_action_steps"]) * joint_count
        ),
    }


def package_graph(source: Path, sizes: dict[str, int]) -> onnx.ModelProto:
    source_model = onnx.load(str(source), load_external_data=True)
    if len(source_model.graph.input) != 1:
        raise ValueError("Expected one flat G1 PMG source input")
    source_input = source_model.graph.input[0]
    shape = [dim.dim_value for dim in source_input.type.tensor_type.shape.dim]
    if source_input.name != "policy" or shape != [1, sum(sizes.values())]:
        raise ValueError(f"Unexpected source policy input: {source_input.name} {shape}")
    if "action" not in {output.name for output in source_model.graph.output}:
        raise ValueError("Source graph must expose its deterministic action tensor")

    # In the release graph, action is the actor mean (metadata calls it loc);
    # action_alias_6 is the identical conceptual action. Retain only its ancestors.
    model = onnx.utils.Extractor(source_model).extract_model(["policy"], ["action"])
    del model.graph.input[:]
    model.graph.input.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width])
        for name, width in sizes.items()
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "Concat", list(sizes), ["policy"], axis=1, name="heft_semantic_inputs"
        ),
    )
    model.graph.name = "heft_g1_pmg_sim2real"
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    return model


def make_deploy_config(tracking: dict, controller: dict) -> dict[str, Any]:
    names = list(controller["policy_joint_names"])
    if names != list(tracking["action_joint_names"]):
        raise ValueError("Source controller and action orders differ")
    robot_cfg = get_robot_cfg("g1")
    if list(tracking["dataset_joint_names"]) != list(robot_cfg.joint_names):
        raise ValueError("Source dataset joint order differs from sim2real G1 order")
    for field in ("kps", "kds", "default_qpos"):
        if len(controller[field]) != len(names):
            raise ValueError(f"Source controller {field} has the wrong joint count")
    if len(tracking["action_scale"]) != len(names):
        raise ValueError("Source action_scale has the wrong joint count")

    core_args = {
        "future_steps": list(tracking["future_steps"]),
        **{key: list(tracking[key]) for key in HISTORY_KEYS},
        "prev_action_steps": int(tracking["prev_action_steps"]),
        "boot_indicator_max": int(tracking["boot_indicator_max"]),
        "joint_names": names,
        "root_body_name": "pelvis",
    }
    observation = {}
    for component in semantic_sizes(tracking, len(names)):
        # Supply the complete common contract to every view. The runtime shares
        # one core and advances its histories only through the first view.
        observation[component] = {
            component: {
                "_target_": "heft.heft_component_obs",
                "component": component,
                "policy_variant": "pmg",
                **core_args,
            }
        }
    return {
        "model_path": "policy.onnx",
        "clip_actions": float(tracking["action_clip"]),
        "observation": observation,
        "joint_names_simulation": list(tracking["dataset_joint_names"]),
        "body_names_simulation": list(robot_cfg.body_names),
        "policy_joint_names": names,
        "default_joint_pos": dict(zip(names, map(float, controller["default_qpos"]))),
        "joint_kp": dict(zip(names, map(float, controller["kps"]))),
        "joint_kd": dict(zip(names, map(float, controller["kds"]))),
        "action_scale": list(map(float, tracking["action_scale"])),
        "motion": {
            "motion_backend": "zmq",
            "motion_zmq_connect": "tcp://127.0.0.1:28701",
            "motion_zmq_hwm": 1,
            "motion_dt_s": 1.0 / float(controller["control_freq"]),
            "motion_tolerance_s": 0.04,
            "future_steps": list(tracking["future_steps"]),
            "joint_names": names,
            "body_names": ["pelvis"],
            "root_body_name": "pelvis",
            "anchor_body_name": "pelvis",
        },
    }


def realistic_feed(rng: np.random.Generator, tracking: dict, controller: dict) -> dict:
    """Nonzero, correlated robot/reference states, including valid rotations."""
    dt = 1.0 / float(controller["control_freq"])
    time = np.asarray(tracking["future_steps"], dtype=float) * dt
    default = np.asarray(controller["default_qpos"], dtype=float)
    joint_count = len(default)
    root_rotvec = rng.normal(0.0, [0.15, 0.15, 0.7])
    ref_rotation = Rotation.from_rotvec(
        root_rotvec + time[:, None] * rng.normal(0.0, 0.4, (1, 3))
    )
    actual_rotation = Rotation.from_rotvec(root_rotvec + rng.normal(0.0, 0.08, 3))
    root_position = rng.normal(0.0, 0.1, (1, 3)) + [0.0, 0.0, 0.78]
    root_position = root_position + time[:, None] * rng.normal(0.0, [0.5, 0.3, 0.1], (1, 3))
    displacements = ref_rotation[0].inv().apply(root_position[1:] - root_position[0])
    relative_rotation = (actual_rotation.inv() * ref_rotation).as_matrix()
    current_joints = default + rng.normal(0.0, 0.12, joint_count)
    reference_joints = (
        current_joints + rng.normal(0.0, 0.08, joint_count)
        + time[:, None] * rng.normal(0.0, 0.6, (1, joint_count))
    )
    history_parts = []
    for key in HISTORY_KEYS:
        history_time = -np.asarray(tracking[key], dtype=float) * dt
        if key == "root_angvel_history_steps":
            history = rng.normal(0.0, 0.5, (1, 3)) + history_time[:, None] * rng.normal(0.0, 0.1, (1, 3))
        elif key == "projected_gravity_history_steps":
            history_rotation = Rotation.from_rotvec(
                actual_rotation.as_rotvec() + history_time[:, None] * rng.normal(0.0, 0.2, (1, 3))
            )
            history = history_rotation.inv().apply([0.0, 0.0, -1.0])
        elif key == "joint_pos_history_steps":
            history = current_joints + history_time[:, None] * rng.normal(0.0, 0.3, (1, joint_count))
        else:
            history = rng.normal(0.0, 0.6, (len(history_time), joint_count))
        history_parts.append(history.reshape(-1))
    history_parts.append(rng.normal(0.0, 0.8, (int(tracking["prev_action_steps"]), joint_count)).reshape(-1))
    boot_max = int(tracking["boot_indicator_max"])
    feed = {
        "context": np.array([rng.integers(0, boot_max + 1) / boot_max]),
        "motion_command": np.concatenate([
            displacements.reshape(-1),
            relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(-1),
        ]),
        "target_motion": np.concatenate([
            reference_joints.reshape(-1),
            (reference_joints - current_joints).reshape(-1),
            root_position[:, 2],
            ref_rotation.inv().apply([0.0, 0.0, -1.0]).reshape(-1),
        ]),
        "proprioception": np.concatenate(history_parts),
    }
    return {name: value[None, :].astype(np.float32) for name, value in feed.items()}


def validate(source: Path, target: Path, tracking: dict, controller: dict, runs: int, seed: int) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    source_session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    target_session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    sizes = semantic_sizes(tracking, len(controller["policy_joint_names"]))
    actual_inputs = {item.name: item.shape for item in target_session.get_inputs()}
    if actual_inputs != {name: [1, width] for name, width in sizes.items()}:
        raise ValueError(f"Unexpected exported input contract: {actual_inputs}")
    if [(item.name, item.shape) for item in target_session.get_outputs()] != [("action", [1, 29])]:
        raise ValueError("Exported action contract must be action[1,29]")
    max_error = 0.0
    max_repeat_error = 0.0
    max_source_alias_error = 0.0
    rng = np.random.default_rng(seed)
    source_names = [item.name for item in source_session.get_outputs()]
    for _ in range(runs):
        feed = realistic_feed(rng, tracking, controller)
        flat = np.concatenate([feed[name] for name in sizes], axis=-1)
        original = dict(zip(source_names, source_session.run(None, {"policy": flat})))
        actual = target_session.run(["action"], feed)[0]
        repeated = target_session.run(["action"], feed)[0]
        if not np.isfinite(actual).all():
            raise ValueError("Export produced nonfinite actions")
        error = float(np.max(np.abs(actual - original["action"])))
        max_error = max(max_error, error)
        max_repeat_error = max(max_repeat_error, float(np.max(np.abs(actual - repeated))))
        np.testing.assert_allclose(actual, original["action"], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(actual, repeated)
        if "action_alias_6" in original:
            alias_error = float(np.max(np.abs(original["action"] - original["action_alias_6"])))
            max_source_alias_error = max(max_source_alias_error, alias_error)
            np.testing.assert_array_equal(original["action"], original["action_alias_6"])
    return {
        "runs": runs, "seed": seed, "provider": "CPUExecutionProvider",
        "inputs": "correlated nonzero joint/root states with valid rotations and histories",
        "action_max_abs_error": max_error,
        "repeat_action_max_abs_error": max_repeat_error,
        "source_action_alias_max_abs_error": max_source_alias_error,
        "source_output_names": source_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Extracted motion_tracking sim2real branch root",
    )
    parser.add_argument("--source-commit", default=SOURCE_DEPLOY_COMMIT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints/heft/g1_pmg")
    parser.add_argument("--compare-runs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.compare_runs < 32:
        parser.error("--compare-runs must be at least 32")
    source_root = args.source_root.resolve()
    cfg_dir = source_root / "sim2real/config/g1"
    tracking_path = cfg_dir / "tracking.yaml"
    controller_path = cfg_dir / "controller.yaml"
    tracking = yaml.safe_load(tracking_path.read_text())
    controller = yaml.safe_load(controller_path.read_text())
    if tracking.get("use_compliance_flag_obs", False):
        raise ValueError("This exporter packages G1 PMG; compliance observations are unsupported")
    source_model = (cfg_dir / tracking["policy_path"]).resolve()
    if source_model.parent.name != "G1_PMG":
        raise ValueError(f"Expected released G1_PMG policy, got {source_model}")
    sizes = semantic_sizes(tracking, len(controller["policy_joint_names"]))
    model = package_graph(source_model, sizes)
    config = make_deploy_config(tracking, controller)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "policy.onnx"
    onnx.save_model(model, str(target), save_as_external_data=False)
    validation = validate(source_model, target, tracking, controller, args.compare_runs, args.seed)
    class PlainDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True
    (output_dir / "policy.yaml").write_text(yaml.dump(config, Dumper=PlainDumper, sort_keys=False))
    source_files = [
        tracking_path, controller_path, source_model,
        source_model.with_suffix(".json"), Path(str(source_model) + ".data"),
        source_root / "sim2real/src/runtime/observation.py",
        source_root / "sim2real/src/runtime/policy.py",
    ]
    metadata = {
        "in_keys": list(sizes), "out_keys": ["action"],
        "in_shapes": [[1, width] for width in sizes.values()],
        "out_shapes": [[1, 29]],
        "source": {
            "repository": "https://github.com/Axellwppr/motion_tracking",
            "branch": "sim2real", "commit": args.source_commit,
            "files_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
            "observation_implementation": "sim2real/src/runtime/observation.py",
            "observation_order": "sim2real/src/runtime/policy.py:TrackingPolicyRaw._build_obs_modules",
        },
        "export": {
            "script": "scripts/export_heft_policy.py", "onnx_sha256": sha256(target),
            "semantic_input_sizes": sizes, "normalization": "embedded unchanged in source graph",
            "action_output": "deterministic actor mean; equals source conceptual action alias",
            "single_file": True, "ir_version": model.ir_version,
            "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        },
        "validation": validation,
    }
    (output_dir / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "semantic_input_sizes": sizes,
        "validation": validation,
    }, indent=2))


if __name__ == "__main__":
    main()
