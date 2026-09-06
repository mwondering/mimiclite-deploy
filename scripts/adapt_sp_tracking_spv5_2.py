#!/usr/bin/env python3
"""Adapt an SP_Tracking SPV5-2 export to the sim2real runtime contract.

The source actor already contains the complete reference encoder and policy.
This adapter preserves every learned node and only replaces its flat 8199-D
input with four named semantic inputs.  It also writes the matching deploy
YAML, ONNX sidecar, and provenance manifest used by this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.observations.sp_tracking import (
    SPV5_2_ESTIMATOR_HISTORY_DIM,
    SPV5_2_HISTORY_LENGTH,
    SPV5_2_JOINT_NAMES,
    SPV5_2_KEY_BODY_DIM,
    SPV5_2_REFERENCE_INPUT_DIM,
    SPV5_2_REFERENCE_STEPS,
)


SOURCE_INPUT_NAME = "spv5_2_observation"
SOURCE_CODEBASE = "SP_Tracking"
POLICY_VARIANT = "SPV5-2Actor"
SEMANTIC_INPUTS = (
    ("robot_root_quat", 4),
    ("estimator_history", SPV5_2_ESTIMATOR_HISTORY_DIM),
    ("reference_encoder_input", SPV5_2_REFERENCE_INPUT_DIM),
    ("robot_key_body", SPV5_2_KEY_BODY_DIM),
)
OUTPUT_NAME = "action"
SOURCE_INPUT_DIM = sum(dim for _, dim in SEMANTIC_INPUTS)
ACTION_DIM = len(SPV5_2_JOINT_NAMES)

DEFAULT_JOINT_POS = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)

KP_5020 = 14.25062309787429
KD_5020 = 0.907222843292423
KP_7520_14 = 40.17923863450712
KD_7520_14 = 2.557889775413375
KP_7520_22 = 99.09842777666111
KD_7520_22 = 6.308801853496639
KP_4010 = 16.77832748089279
KD_4010 = 1.06814150219
KP_PARALLEL_5020 = 28.50124619574858
KD_PARALLEL_5020 = 1.814445686584846

JOINT_KP = (
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_PARALLEL_5020,
    KP_PARALLEL_5020,
    KP_7520_14,
    KP_7520_22,
    KP_7520_14,
    KP_7520_22,
    KP_PARALLEL_5020,
    KP_PARALLEL_5020,
    KP_7520_14,
    KP_PARALLEL_5020,
    KP_PARALLEL_5020,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_4010,
    KP_4010,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_5020,
    KP_4010,
    KP_4010,
)
JOINT_KD = (
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_PARALLEL_5020,
    KD_PARALLEL_5020,
    KD_7520_14,
    KD_7520_22,
    KD_7520_14,
    KD_7520_22,
    KD_PARALLEL_5020,
    KD_PARALLEL_5020,
    KD_7520_14,
    KD_PARALLEL_5020,
    KD_PARALLEL_5020,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_4010,
    KD_4010,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_5020,
    KD_4010,
    KD_4010,
)

SCALE_5020 = 0.43857731392336724
SCALE_7520_14 = 0.5475464629911068
SCALE_7520_22 = 0.35066146637882434
SCALE_4010 = 0.07450087032950714
ACTION_SCALE = (
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_5020,
    SCALE_5020,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_7520_14,
    SCALE_7520_22,
    SCALE_5020,
    SCALE_5020,
    SCALE_7520_14,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_4010,
    SCALE_4010,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_5020,
    SCALE_4010,
    SCALE_4010,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing either policy_<iteration>.onnx/.json or "
            "policy.onnx/config.yaml, plus checkpoint_<iteration>.pt."
        ),
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration to select when the directory contains more than one export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/sp-tracking/spv5_2"),
    )
    parser.add_argument(
        "--equivalence-trials",
        type=int,
        default=5,
        help="Random CPU ONNX Runtime comparisons; set to 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _select_source_files(
    checkpoint_dir: Path,
    iteration: int | None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = checkpoint_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")

    sidecars = sorted(root.glob("policy_*.json"))
    if iteration is not None:
        sidecars = [path for path in sidecars if path.stem == f"policy_{iteration}"]
    if len(sidecars) > 1:
        raise ValueError(
            f"Expected at most one source sidecar, found {[path.name for path in sidecars]}; "
            "pass --iteration to disambiguate"
        )

    if sidecars:
        metadata_path = sidecars[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_iteration = int(metadata["iteration"])
        if iteration is not None and selected_iteration != iteration:
            raise ValueError(
                f"Sidecar iteration {selected_iteration} does not match "
                f"--iteration {iteration}"
            )
        source_onnx = root / f"policy_{selected_iteration}.onnx"
        expected_checkpoint = f"checkpoint_{selected_iteration}.pt"
        checkpoint_name = str(metadata.get("checkpoint", expected_checkpoint))
        if checkpoint_name != expected_checkpoint:
            raise ValueError(
                f"Sidecar checkpoint {checkpoint_name!r} does not match iteration "
                f"{selected_iteration}; expected {expected_checkpoint!r}"
            )
        source_pt = root / expected_checkpoint
    else:
        metadata_path = root / "config.yaml"
        source_onnx = root / "policy.onnx"
        checkpoints = sorted(root.glob("checkpoint_*.pt"))
        if iteration is not None:
            checkpoints = [
                path for path in checkpoints if path.stem == f"checkpoint_{iteration}"
            ]
        if len(checkpoints) != 1:
            raise ValueError(
                "A sidecar-free export requires exactly one checkpoint_<iteration>.pt; "
                f"found {[path.name for path in checkpoints]}"
            )
        source_pt = checkpoints[0]
        try:
            selected_iteration = int(source_pt.stem.removeprefix("checkpoint_"))
        except ValueError as exc:
            raise ValueError(
                f"Cannot infer iteration from checkpoint name {source_pt.name!r}"
            ) from exc
        metadata = _metadata_from_resolved_config(metadata_path, selected_iteration)

    for path in (source_onnx, metadata_path, source_pt):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")
    return source_onnx, metadata_path, source_pt, metadata


def _metadata_from_resolved_config(
    config_path: Path,
    iteration: int,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            "The export has no policy_<iteration>.json sidecar and no resolved "
            f"training config: {config_path}"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = config.get("task", {})
    agent_overrides = task.get("agent_overrides", {})
    actor = agent_overrides.get("actor", {})
    observations = task.get("obs", {}).get("observations", {})

    expected_actor_groups = [name for name, _ in SEMANTIC_INPUTS]
    actor_groups = agent_overrides.get("obs_groups", {}).get("actor")
    if actor_groups != expected_actor_groups:
        raise ValueError(
            "Resolved config actor observation groups do not match SPV5-2: "
            f"expected={expected_actor_groups}, got={actor_groups}"
        )

    expected_history_terms = (
        "joint_pos",
        "joint_vel",
        "projected_gravity",
        "base_ang_vel",
        "last_action",
        "joint_torque",
    )
    history_terms = observations.get("estimator_history", {}).get("terms", [])
    history_names = tuple(term.get("name") for term in history_terms)
    history_lengths = tuple(term.get("history_length") for term in history_terms)
    if history_names != expected_history_terms or history_lengths != (
        SPV5_2_HISTORY_LENGTH,
    ) * len(expected_history_terms):
        raise ValueError(
            "Resolved config estimator history does not match SPV5-2: "
            f"names={history_names}, lengths={history_lengths}"
        )

    expected_observation_terms = {
        "robot_root_quat": "spv5_robot_root_quat",
        "reference_encoder_input": "spv5_reference_encoder_input",
        "robot_key_body": "spv5_2_robot_key_body_state",
    }
    for group_name, expected_term in expected_observation_terms.items():
        terms = observations.get(group_name, {}).get("terms", [])
        actual_terms = [term.get("term") for term in terms]
        if actual_terms != [expected_term]:
            raise ValueError(
                f"Resolved config group {group_name!r} must contain "
                f"{expected_term!r}, got {actual_terms}"
            )

    actor_class_name = str(actor.get("class_name", ""))
    if actor_class_name.rsplit(":", 1)[-1] not in {
        "SPV52Actor",
        "SPV52HeightContactEstimatorActor",
    }:
        raise ValueError(
            f"Unsupported SPV5-2 actor class in resolved config: {actor_class_name!r}"
        )
    if actor.get("estimator_history_length") != SPV5_2_HISTORY_LENGTH:
        raise ValueError(
            "Resolved config estimator_history_length must be "
            f"{SPV5_2_HISTORY_LENGTH}, got {actor.get('estimator_history_length')}"
        )
    if float(actor.get("reference_fps", 0.0)) != 50.0:
        raise ValueError(
            f"Resolved config reference_fps must be 50, got {actor.get('reference_fps')}"
        )

    robot = task.get("robot", {})
    action = task.get("action", {})
    sim = task.get("sim", {})
    control_dt = float(sim.get("timestep", 0.0)) * int(task.get("decimation", 0))
    if robot.get("name") != "g1" or robot.get("anchor_body_name") != "pelvis":
        raise ValueError(
            "Resolved config must use the G1 pelvis contract; "
            f"robot={robot.get('name')!r}, anchor={robot.get('anchor_body_name')!r}"
        )
    if action.get("scale") != "g1_action_scale" or not action.get(
        "use_default_offset", False
    ):
        raise ValueError(f"Resolved config action contract is unsupported: {action}")
    if abs(control_dt - 0.02) > 1.0e-12:
        raise ValueError(f"Resolved config control timestep must be 0.02 s, got {control_dt}")

    configured_checkpoint = str(config.get("checkpoint_path", ""))
    run_name = Path(configured_checkpoint).parent.name if configured_checkpoint else ""
    if not run_name:
        run_name = str(config.get("agent", {}).get("run_name", ""))
    return {
        "format": "motion_tracking_sim2real_resolved_config",
        "run_name": run_name,
        "iteration": int(iteration),
        "checkpoint": f"checkpoint_{iteration}.pt",
        "configured_checkpoint_path": configured_checkpoint,
        "in_keys": [SOURCE_INPUT_NAME],
        "out_keys": [OUTPUT_NAME],
        "num_actions": ACTION_DIM,
        "anchor_body_name": "pelvis",
        "body_names": list(robot.get("body_names", [])),
        "policy_variant": actor_class_name.rsplit(":", 1)[-1],
        "observation_profile": task.get("variant", {}).get("observation_profile"),
        "control_dt_s": control_dt,
    }


def _validate_source_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "in_keys": [SOURCE_INPUT_NAME],
        "out_keys": [OUTPUT_NAME],
        "num_actions": ACTION_DIM,
        "anchor_body_name": "pelvis",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Source metadata is not an SPV5-2 actor export: {mismatches}")
    supported_formats = {
        "motion_tracking_sim2real_policy",
        "motion_tracking_sim2real_resolved_config",
    }
    if metadata.get("format") not in supported_formats:
        raise ValueError(
            f"Unsupported source metadata format {metadata.get('format')!r}; "
            f"expected one of {sorted(supported_formats)}"
        )


def _rewrite_with_semantic_inputs(source_path: Path, output_path: Path) -> None:
    model = onnx.load(source_path)
    if len(model.graph.input) != 1 or model.graph.input[0].name != SOURCE_INPUT_NAME:
        raise ValueError(
            "Expected one ONNX input named "
            f"{SOURCE_INPUT_NAME!r}, got {[value.name for value in model.graph.input]}"
        )
    if len(model.graph.output) != 1 or model.graph.output[0].name != OUTPUT_NAME:
        raise ValueError(
            f"Expected one ONNX output named {OUTPUT_NAME!r}, "
            f"got {[value.name for value in model.graph.output]}"
        )

    source_input = model.graph.input[0]
    source_shape = _shape(source_input)
    output_shape = _shape(model.graph.output[0])
    if source_shape != [1, SOURCE_INPUT_DIM]:
        raise ValueError(
            f"Expected source input shape [1, {SOURCE_INPUT_DIM}], got {source_shape}"
        )
    if output_shape != [1, ACTION_DIM]:
        raise ValueError(f"Expected action shape [1, {ACTION_DIM}], got {output_shape}")
    if source_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("SPV5-2 source input must be float32")
    if model.graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("SPV5-2 action output must be float32")

    semantic_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, dim])
        for name, dim in SEMANTIC_INPUTS
    ]
    del model.graph.input[:]
    model.graph.input.extend(semantic_inputs)
    model.graph.node.insert(
        0,
        helper.make_node(
            "Concat",
            [name for name, _ in SEMANTIC_INPUTS],
            [SOURCE_INPUT_NAME],
            axis=1,
            name=f"assemble_{SOURCE_INPUT_NAME}",
        ),
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def _ordered_joint_values(values: Sequence[float]) -> dict[str, float]:
    if len(values) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} joint values, got {len(values)}")
    return {
        name: float(value)
        for name, value in zip(SPV5_2_JOINT_NAMES, values, strict=True)
    }


def _policy_yaml() -> dict[str, Any]:
    robot_cfg = get_robot_cfg("g1")
    observations = {
        name: {
            name: {
                "_target_": f"sp_tracking.spv5_2_{name}",
            }
        }
        for name, _ in SEMANTIC_INPUTS
    }
    return {
        "model_path": "policy.onnx",
        "observation": observations,
        "joint_names_simulation": list(robot_cfg.joint_names),
        "body_names_simulation": list(robot_cfg.body_names),
        "policy_joint_names": list(SPV5_2_JOINT_NAMES),
        "default_joint_pos": _ordered_joint_values(DEFAULT_JOINT_POS),
        "joint_kp": _ordered_joint_values(JOINT_KP),
        "joint_kd": _ordered_joint_values(JOINT_KD),
        "action_scale": _ordered_joint_values(ACTION_SCALE),
        "clip_actions": 10.0,
        "motion": {
            "motion_backend": "npz",
            "motion_path": (
                "hf://elijahgalahad/any4hdmi-g1-lafan/"
                "motions/walk1_subject1.npz"
            ),
            "future_steps": list(SPV5_2_REFERENCE_STEPS),
            "root_body_name": "pelvis",
            "motion_dt_s": 0.02,
        },
    }


def _semantic_sidecar(
    source_metadata: dict[str, Any],
    source_files: dict[str, dict[str, Any]],
    adapted_onnx_sha256: str,
) -> dict[str, Any]:
    return {
        "format": "sim2real_semantic_policy",
        "source_codebase": SOURCE_CODEBASE,
        "policy_variant": source_metadata.get("policy_variant", POLICY_VARIANT),
        "source_format": source_metadata.get("format"),
        "run_name": source_metadata.get("run_name"),
        "iteration": int(source_metadata["iteration"]),
        "checkpoint": source_metadata.get("checkpoint"),
        "in_keys": [name for name, _ in SEMANTIC_INPUTS],
        "out_keys": [OUTPUT_NAME],
        "in_shapes": [[1, dim] for _, dim in SEMANTIC_INPUTS],
        "out_shapes": [[1, ACTION_DIM]],
        "num_actions": ACTION_DIM,
        "observation_contract": {
            "flat_order": [name for name, _ in SEMANTIC_INPUTS],
            "estimator_history": {
                "length": 50,
                "layout": "term-major; each term oldest-to-newest",
                "terms": [
                    ["joint_pos_relative_to_default", 29],
                    ["joint_velocity", 29],
                    ["projected_gravity", 3],
                    ["root_angular_velocity_body", 3],
                    ["previous_raw_action", 29],
                    ["joint_torque", 29],
                ],
                "reset": "backfill all slots with the first measured sample",
            },
            "reference_encoder_input": {
                "steps": list(SPV5_2_REFERENCE_STEPS),
                "fps": 50.0,
                "frame_layout": ["pelvis_position_world", "pelvis_rotation_6d", "joint_position"],
                "rotation_6d": "first two matrix columns, each column contiguous",
            },
            "robot_key_body": {
                "count": 13,
                "layout": "field-major: position, rotation_6d, linear_velocity, root_relative_angular_velocity",
                "kinematics_asset": (
                    "sim2real/rl_policy/observations/assets/"
                    "g1_sp_tracking_kinematics.xml"
                ),
            },
            "deploy_observation_noise": False,
        },
        "adaptation": {
            "type": "semantic_input_concat_wrapper",
            "learned_nodes_or_weights_changed": False,
            "adapted_onnx_sha256": adapted_onnx_sha256,
            "source_files": source_files,
        },
    }


def _compare_onnx(
    source_path: Path,
    adapted_path: Path,
    *,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    if trials <= 0:
        return {"trials": 0, "max_abs_error": 0.0, "mean_abs_error": 0.0}
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for equivalence checks; install "
            "--extra inference-cpu or pass --equivalence-trials 0"
        ) from exc

    providers = ["CPUExecutionProvider"]
    source = ort.InferenceSession(str(source_path), providers=providers)
    adapted = ort.InferenceSession(str(adapted_path), providers=providers)
    rng = np.random.default_rng(seed)
    max_abs_error = 0.0
    total_abs_error = 0.0
    total_values = 0
    for _ in range(trials):
        semantic = {
            name: rng.standard_normal((1, dim), dtype=np.float32)
            for name, dim in SEMANTIC_INPUTS
        }
        flat = np.concatenate([semantic[name] for name, _ in SEMANTIC_INPUTS], axis=1)
        expected = source.run(None, {SOURCE_INPUT_NAME: flat})[0]
        actual = adapted.run(None, semantic)[0]
        if not np.array_equal(actual, expected):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        error = np.abs(actual - expected)
        max_abs_error = max(max_abs_error, float(error.max(initial=0.0)))
        total_abs_error += float(error.sum())
        total_values += int(error.size)
    return {
        "trials": trials,
        "max_abs_error": max_abs_error,
        "mean_abs_error": total_abs_error / max(total_values, 1),
    }


def main() -> None:
    args = _parse_args()
    source_onnx, source_metadata_path, source_pt, source_metadata = _select_source_files(
        args.checkpoint_dir,
        args.iteration,
    )
    _validate_source_metadata(source_metadata)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = output_dir / "policy.onnx"
    if output_onnx.resolve() == source_onnx.resolve():
        raise ValueError("Output policy.onnx must not overwrite the source ONNX")

    _rewrite_with_semantic_inputs(source_onnx, output_onnx)
    equivalence = _compare_onnx(
        source_onnx,
        output_onnx,
        trials=args.equivalence_trials,
        seed=args.seed,
    )
    source_files = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (source_onnx, source_metadata_path, source_pt)
    }
    sidecar = _semantic_sidecar(source_metadata, source_files, _sha256(output_onnx))
    sidecar["adaptation"]["equivalence"] = equivalence
    (output_dir / "policy.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "policy.yaml").write_text(
        yaml.safe_dump(_policy_yaml(), sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# SP_Tracking SPV5-2 sim2real artifact",
                "",
                f"Source run: `{source_metadata.get('run_name')}`",
                f"Source iteration: `{source_metadata['iteration']}`",
                "",
                "`policy.onnx` preserves the source actor and adds four named semantic inputs.",
                "The training `.pt` file is provenance-only and is not copied into this runtime artifact.",
                "",
            )
        ),
        encoding="utf-8",
    )

    (output_dir / "README_zh.md").write_text(
        "\n".join(
            (
                "# SP_Tracking SPV5-2 部署文件",
                "",
                f"源训练：`{source_metadata.get('run_name')}`",
                f"源训练步数：`{source_metadata['iteration']}`",
                "",
                "`policy.onnx` 保留完整源 actor，并增加四个语义输入。",
                "训练 `.pt` 仅用于来源核验，不会复制到部署目录。",
                "",
            )
        ),
        encoding="utf-8",
    )

    print(f"source={source_onnx}")
    print(f"output={output_onnx}")
    print(f"inputs={list(SEMANTIC_INPUTS)} output=({OUTPUT_NAME!r}, {ACTION_DIM})")
    print(f"equivalence={equivalence}")


if __name__ == "__main__":
    main()
