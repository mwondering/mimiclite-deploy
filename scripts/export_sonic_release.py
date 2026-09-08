#!/usr/bin/env python3
"""Merge NVIDIA's default SONIC release into semantic-input sim2real models.

The release universal encoder has a 1762-value interface, including obsolete
height fields. This is different from the low-latency release. The wrapper
zero-fills inactive fields and fixes mode 0 (G1) or 2 (SMPL); both learned graphs,
including finite scalar quantization, are copied without changing operations.
Inputs retain the deployer's group-flat layout, not the intermediate MLP layout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper
import onnxruntime as ort
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOSITORY = "nvidia/GEAR-SONIC"
MODEL_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
DEPLOY_COMMIT = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38"
SOURCE_HASHES = {
    "model_encoder.onnx": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "model_decoder.onnx": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
    "observation_config.yaml": "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
}
# [start, stop) offsets in the released encoder's 1762-value obs_dict.
UNIVERSAL_FIELDS = {
    "encoder_mode_4": [0, 4],
    "motion_joint_positions_10frame_step5": [4, 294],
    "motion_joint_velocities_10frame_step5": [294, 584],
    "motion_root_z_position_10frame_step5": [584, 594],
    "motion_root_z_position": [594, 595],
    "motion_anchor_orientation": [595, 601],
    "motion_anchor_orientation_10frame_step5": [601, 661],
    "motion_joint_positions_lowerbody_10frame_step5": [661, 781],
    "motion_joint_velocities_lowerbody_10frame_step5": [781, 901],
    "vr_3point_local_target": [901, 910],
    "vr_3point_local_orn_target": [910, 922],
    "smpl_joints_10frame_step1": [922, 1642],
    "smpl_anchor_orientation_10frame_step1": [1642, 1702],
    "motion_joint_positions_wrists_10frame_step1": [1702, 1762],
}
MODE_DIMENSIONS = {"g1": 640, "smpl": 840}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(source_root: Path) -> None:
    for name, expected in SOURCE_HASHES.items():
        actual = sha256(source_root / name)
        if actual != expected:
            raise ValueError(
                f"{name} SHA256 {actual} differs from the pinned default release; "
                "re-audit the universal field layout before exporting other weights"
            )


def _float_info(name: str, size: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [size])


def package_graph(source_root: Path, mode: str) -> onnx.ModelProto:
    encoder = compose.add_prefix(onnx.load(source_root / "model_encoder.onnx"), "encoder/")
    decoder = compose.add_prefix(onnx.load(source_root / "model_decoder.onnx"), "decoder/")
    for model, expected_input, expected_output in ((encoder, 1762, 64), (decoder, 994, 29)):
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("Expected exactly one input and one output per released graph")
        for value, width in ((model.graph.input[0], expected_input), (model.graph.output[0], expected_output)):
            if [d.dim_value for d in value.type.tensor_type.shape.dim] != [1, width]:
                raise ValueError(f"Unexpected source tensor contract: {value}")
    if list(encoder.opset_import) != list(decoder.opset_import):
        raise ValueError("Source encoder and decoder opsets differ")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    def const(name: str, value: np.ndarray) -> str:
        name = f"wrapper/{name}"
        initializers.append(numpy_helper.from_array(value, name=name))
        return name

    def zeros(name: str, count: int) -> str:
        return const(name, np.zeros((1, count), dtype=np.float32))

    semantic_name = f"{mode}_input"
    batch_axis = const("batch_axis", np.array([0], dtype=np.int64))
    batched_semantic = "wrapper/semantic_batched"
    batched_proprioception = "wrapper/proprioception_batched"
    nodes.append(helper.make_node("Unsqueeze", [semantic_name, batch_axis], [batched_semantic]))
    nodes.append(helper.make_node("Unsqueeze", ["proprioception", batch_axis], [batched_proprioception]))
    if mode == "g1":
        axes = const("slice_axes", np.array([1], dtype=np.int64))
        steps = const("slice_steps", np.array([1], dtype=np.int64))
        for name, start, stop in (("g1_command", 0, 580), ("g1_orientation", 580, 640)):
            nodes.append(helper.make_node(
                "Slice", [batched_semantic, const(f"{name}_start", np.array([start], dtype=np.int64)),
                          const(f"{name}_stop", np.array([stop], dtype=np.int64)), axes, steps],
                [f"wrapper/{name}"], name=f"wrapper/{name}",
            ))
        parts = [zeros("g1_mode", 4), "wrapper/g1_command", zeros("unused_heights_anchor", 17),
                 "wrapper/g1_orientation", zeros("inactive_teleop_smpl", 1101)]
    elif mode == "smpl":
        header = np.zeros((1, 922), dtype=np.float32)
        header[0, 0] = 2.0
        parts = [const("smpl_mode_and_inactive_fields", header), batched_semantic]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    nodes.append(helper.make_node("Concat", parts, [encoder.graph.input[0].name], axis=1, name="wrapper/encoder_input"))
    nodes.extend(deepcopy(encoder.graph.node))
    nodes.append(helper.make_node(
        "Concat", [encoder.graph.output[0].name, batched_proprioception], [decoder.graph.input[0].name],
        axis=1, name="wrapper/decoder_input",
    ))
    nodes.extend(deepcopy(decoder.graph.node))
    nodes.append(helper.make_node("Squeeze", [decoder.graph.output[0].name, batch_axis], ["action"], name="wrapper/action"))
    nodes.append(helper.make_node("Squeeze", [encoder.graph.output[0].name, batch_axis], ["token"], name="wrapper/token"))
    graph = helper.make_graph(
        nodes, f"sonic_default_release_{mode}",
        [_float_info(semantic_name, MODE_DIMENSIONS[mode]), _float_info("proprioception", 930)],
        [_float_info("action", 29), _float_info("token", 64)],
        initializer=[*initializers, *deepcopy(encoder.graph.initializer), *deepcopy(decoder.graph.initializer)],
        value_info=[*deepcopy(encoder.graph.value_info), *deepcopy(decoder.graph.value_info)],
    )
    model = helper.make_model(graph, producer_name="sim2real.export_sonic_release", opset_imports=encoder.opset_import)
    model.ir_version = max(encoder.ir_version, decoder.ir_version)
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    return model


def universal_input(mode: str, semantic: np.ndarray) -> np.ndarray:
    """Independent NumPy packing for comparison with the untouched source files."""
    universal = np.zeros((1, 1762), dtype=np.float32)
    if mode == "g1":
        universal[:, 4:584] = semantic[:, :580]
        universal[:, 601:661] = semantic[:, 580:]
    else:
        universal[:, 0] = 2.0
        universal[:, 922:1762] = semantic
    return universal


def structured_feed(rng: np.random.Generator, mode: str, run: int) -> dict[str, np.ndarray]:
    """Correlated joint/point trajectories, rotation matrices and history blocks.

    This tests graph packaging, not simulated tracking. The independent runtime
    observation tests and integrated simulation check the physical conventions.
    """
    dt = 0.1 if mode == "g1" else 0.02
    time_s = np.arange(10) * dt
    phase = rng.uniform(-np.pi, np.pi, 29)
    amplitude = rng.uniform(0.02, 0.4, 29)
    frequency = rng.uniform(0.1, 1.0, 29)
    positions = amplitude * np.sin(2 * np.pi * frequency * time_s[:, None] + phase)
    velocities = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * time_s[:, None] + phase)
    rotations = Rotation.from_rotvec(rng.normal(0, 0.12, (1, 3)) + time_s[:, None] * rng.normal(0, 0.15, (1, 3)))
    orientation = rotations.as_matrix()[:, :, :2].reshape(10, 6)
    if mode == "g1":
        semantic = np.concatenate((positions.ravel(), velocities.ravel(), orientation.ravel()))
    else:
        # A 24-point pelvis-relative body shape with smooth, nonzero displacements.
        body = rng.normal(0, [0.2, 0.12, 0.35], (24, 3))
        body[0] = 0
        displacements = rng.normal(0, 0.03, (24, 3))
        displacements[0] = 0
        joints = body[None] + np.sin(2 * np.pi * time_s[:, None, None]) * displacements[None]
        semantic = np.concatenate((joints.ravel(), orientation.ravel(), positions[:, -6:].ravel()))
    history_t = np.arange(10) * 0.02
    angvel = rng.normal(0, 0.2, (1, 3)) + history_t[:, None] * rng.normal(0, 0.1, (1, 3))
    hist_pos = positions[0] + history_t[:, None] * velocities[0]
    hist_vel = velocities[0] + history_t[:, None] * rng.normal(0, 0.2, (1, 29))
    actions = rng.normal(0, 0.5, (1, 29)) + history_t[:, None] * rng.normal(0, 0.3, (1, 29))
    gravity = Rotation.from_rotvec(history_t[:, None] * rng.normal(0, 0.12, (1, 3))).inv().apply([0, 0, -1])
    histories = [angvel, hist_pos, hist_vel, actions, gravity]
    # Include cold-start zero padding, partially populated histories, and steady state.
    populated = min(run % 13 + 1, 10)
    for history in histories:
        history[:-populated] = 0
    proprioception = np.concatenate([history.ravel() for history in histories])
    return {f"{mode}_input": semantic[None].astype(np.float32), "proprioception": proprioception[None].astype(np.float32)}


def validate(source_root: Path, target: Path, mode: str, runs: int, seed: int) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    encoder = ort.InferenceSession(str(source_root / "model_encoder.onnx"), options, providers=providers)
    decoder = ort.InferenceSession(str(source_root / "model_decoder.onnx"), options, providers=providers)
    merged = ort.InferenceSession(str(target), options, providers=providers)
    assert {i.name: i.shape for i in merged.get_inputs()} == {f"{mode}_input": [MODE_DIMENSIONS[mode]], "proprioception": [930]}
    assert {i.name: i.shape for i in merged.get_outputs()} == {"action": [29], "token": [64]}
    maxima = {"action_max_abs_error": 0.0, "token_max_abs_error": 0.0, "repeat_action_max_abs_error": 0.0}
    rng = np.random.default_rng(seed)
    latency_ms = []
    for run in range(runs):
        feed = structured_feed(rng, mode, run)
        universal = universal_input(mode, feed[f"{mode}_input"])
        token = encoder.run(None, {"obs_dict": universal})[0]
        action = decoder.run(None, {"obs_dict": np.concatenate((token, feed["proprioception"]), axis=1)})[0]
        feed = {name: value[0] for name, value in feed.items()}
        start = time.perf_counter()
        actual_action, actual_token = merged.run(None, feed)
        latency_ms.append((time.perf_counter() - start) * 1000)
        repeat_action, repeat_token = merged.run(None, feed)
        if not np.isfinite(actual_action).all() or not np.isfinite(actual_token).all():
            raise ValueError(f"Nonfinite {mode} output on comparison sample {run}")
        for name, actual, reference in (("action", actual_action, action[0]), ("token", actual_token, token[0])):
            maxima[f"{name}_max_abs_error"] = max(maxima[f"{name}_max_abs_error"], float(np.max(np.abs(actual - reference))))
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5, err_msg=f"{mode} {name}, sample {run}")
        maxima["repeat_action_max_abs_error"] = max(maxima["repeat_action_max_abs_error"], float(np.max(np.abs(actual_action - repeat_action))))
        np.testing.assert_array_equal(actual_action, repeat_action)
        np.testing.assert_array_equal(actual_token, repeat_token)
    return {
        "runs": runs, "seed": seed, "provider": providers[0], "ort_version": ort.__version__,
        "threads": 1, "rtol": 1e-5, "atol": 1e-5,
        "inputs": "nonzero correlated trajectories, valid rotation matrices, and cold/partial/full histories",
        **maxima,
        "merged_inference_median_ms": float(np.median(latency_ms)),
        "merged_inference_p95_ms": float(np.percentile(latency_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "external/groot_sonic_release")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "checkpoints/sonic/release")
    parser.add_argument("--mode", choices=("g1", "smpl", "both"), default="both")
    parser.add_argument("--compare-runs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.compare_runs < 64:
        parser.error("--compare-runs must be at least 64")
    source_root = args.source_root.resolve()
    check_source(source_root)
    modes = MODE_DIMENSIONS if args.mode == "both" else [args.mode]
    for mode in modes:
        output_dir = args.output_root.resolve() / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        model = package_graph(source_root, mode)
        target = output_dir / "policy.onnx"
        onnx.save_model(model, str(target), save_as_external_data=False)
        validation = validate(source_root, target, mode, args.compare_runs, args.seed)
        metadata = {
            "in_keys": [f"{mode}_input", "proprioception"], "out_keys": ["action", "token"],
            "in_shapes": [[MODE_DIMENSIONS[mode]], [930]], "out_shapes": [[29], [64]],
            "source": {
                "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION,
                "deploy_repository": "https://github.com/NVlabs/GR00T-WholeBodyControl",
                "deploy_commit": DEPLOY_COMMIT, "variant": "default", "files_sha256": SOURCE_HASHES,
            },
            "export": {
                "script": "scripts/export_sonic_release.py", "onnx_sha256": sha256(target),
                "mode": mode, "mode_id": 0 if mode == "g1" else 2,
                "single_file": True, "ir_version": model.ir_version,
                "opsets": {i.domain or "ai.onnx": i.version for i in model.opset_import},
                "graph_transform": "untouched universal encoder and decoder; inactive fields zero-filled; fixed mode; semantic input/output wrapper",
                "normalization_and_quantization": "source operations preserved unchanged; no additional observation normalization",
                "semantic_layout": ("joint_positions[10,29] | joint_velocities[10,29] | anchor_orientation[10,6]"
                                    if mode == "g1" else "smpl_joints[10,24,3] | anchor_orientation[10,6] | wrist_positions[10,6]"),
                "proprioception_layout": "angvel[10,3] | joint_pos_minus_default[10,29] | joint_vel[10,29] | last_actions[10,29] | gravity[10,3]",
                "universal_source_fields": UNIVERSAL_FIELDS,
            },
            "validation": validation,
        }
        (output_dir / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"mode": mode, "output": str(target), "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
