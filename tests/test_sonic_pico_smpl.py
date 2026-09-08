"""SONIC PICO preprocessing parity and the initial paused stream contract."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from sim2real.teleop.smpl_stream import (
    DEFAULT_HUMAN_JOINTS_INFO_PATH,
    SMPL_PARENT_INDICES,
    SONIC_WRIST_JOINT_NAMES,
    apply_sonic_wrist_targets,
    build_neutral_smpl_frame,
    build_smpl_frame_from_xrobot_raw,
    sonic_wrist_targets_from_smpl_pose,
)


@pytest.fixture(scope="module")
def human_asset() -> Path:
    path = Path(__file__).resolve().parents[1] / DEFAULT_HUMAN_JOINTS_INFO_PATH
    if not path.exists():
        pytest.skip("SONIC human FK deploy asset is not installed")
    return path


@pytest.fixture(scope="module")
def source_reference(human_asset):
    """Execute the source functions without importing the hardware server."""
    source = Path(os.environ.get(
        "GROOT_WBC_SOURCE",
        str(Path(__file__).resolve().parents[2] / "GR00T-WholeBodyControl"),
    ))
    server = source / "gear_sonic/scripts/pico_manager_thread_server.py"
    if not server.exists():
        pytest.skip("Set GROOT_WBC_SOURCE to run numerical parity against source")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(source))
    try:
        transforms = importlib.import_module("gear_sonic.trl.utils.torch_transform")
        rotations = importlib.import_module("gear_sonic.isaac_utils.rotations")
        conversion = importlib.import_module("gear_sonic.trl.utils.rotation_conversion")
    finally:
        sys.path.remove(str(source))
    transforms.human_joints_info = torch.load(human_asset, map_location="cpu")
    scope = {"np": np, "torch": torch, "sRot": R, "R": R}
    for name in (
        "angle_axis_to_quaternion", "compute_human_joints", "quat_apply",
        "quat_inv", "quaternion_to_angle_axis", "quaternion_to_rotation_matrix",
    ):
        scope[name] = getattr(transforms, name)
    scope.update(
        smpl_root_ytoz_up=rotations.smpl_root_ytoz_up,
        remove_smpl_base_rot=rotations.remove_smpl_base_rot,
        decompose_rotation_aa=conversion.decompose_rotation_aa,
    )
    tree = ast.parse(server.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"process_smpl_joints", "compute_from_body_poses"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(server), "exec"), scope)

    # Use the actual source assignments so the test independently checks wrist
    # signs, decomposition order, Euler convention and Isaac joint indices.
    run_once = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_once"
        and any(isinstance(child, ast.Name) and child.id == "G1_L_WRIST_ROLL_IDX"
                for child in ast.walk(node))
    )
    start = next(i for i, node in enumerate(run_once.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "joint_pos" for t in node.targets))
    end = next(i for i, node in enumerate(run_once.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Name)
                       and t.slice.id == "G1_R_WRIST_YAW_IDX" for t in node.targets))
    wrapper = ast.parse("def source_wrists(use_pose):\n    pass\n").body[0]
    wrapper.body = run_once.body[start:end + 1] + [ast.Return(ast.Name("joint_pos", ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(server), "exec"), scope)
    return scope


def test_raw_pico_smpl_matches_source_numerically(human_asset, source_reference):
    rng = np.random.default_rng(20260908)
    for _ in range(24):
        raw = np.concatenate([rng.normal(size=(24, 3)), R.random(24, random_state=rng).as_quat()], axis=1)
        target = build_smpl_frame_from_xrobot_raw(raw, (), human_joints_info_path=human_asset)
        reference = source_reference["compute_from_body_poses"](
            list(SMPL_PARENT_INDICES), "cpu", raw
        )
        np.testing.assert_allclose(target["smpl_body_pose_aa"],
                                   reference["smpl_pose"].numpy()[0, :63].reshape(21, 3), atol=1e-6)
        np.testing.assert_allclose(target["smpl_joint_pos_root"],
                                   reference["smpl_joints_local"].numpy()[0], atol=5e-6)
        ref_quat = reference["global_orient_quat"].numpy()[0]
        dot = abs(float(np.dot(target["smpl_root_quat_w"], ref_quat)))
        assert dot == pytest.approx(1.0, abs=5e-7)


def test_smpl_wrists_match_actual_source_assignments(source_reference):
    rng = np.random.default_rng(73)
    for _ in range(64):
        body_pose = rng.normal(scale=0.45, size=(21, 3)).astype(np.float32)
        source = source_reference["source_wrists"](body_pose)
        np.testing.assert_allclose(
            sonic_wrist_targets_from_smpl_pose(body_pose),
            source[[23, 24, 25, 26, 27, 28]], atol=3e-7,
        )


def test_neutral_smpl_wrists_are_finite_and_zero():
    np.testing.assert_array_equal(sonic_wrist_targets_from_smpl_pose(np.zeros((21, 3))), np.zeros(6))


def test_smpl_wrist_mapping_preserves_order_and_other_joints():
    names = ["other", *reversed(SONIC_WRIST_JOINT_NAMES)]
    robot = np.arange(len(names), dtype=np.float32)
    body_pose = np.zeros((21, 3), dtype=np.float32)
    body_pose[19] = [0.12, -0.21, 0.05]
    body_pose[20] = [-0.09, 0.18, -0.07]
    mapped = apply_sonic_wrist_targets(robot, names, body_pose)
    expected = sonic_wrist_targets_from_smpl_pose(body_pose)
    np.testing.assert_array_equal(robot, np.arange(len(names)))
    assert mapped[0] == robot[0]
    np.testing.assert_allclose(mapped[[names.index(n) for n in SONIC_WRIST_JOINT_NAMES]], expected)


def test_neutral_smpl_uses_source_fk_with_upright_root(human_asset, source_reference):
    frame = build_neutral_smpl_frame(human_asset)
    torch = source_reference["torch"]
    reference = source_reference["process_smpl_joints"](
        torch.from_numpy(frame["smpl_body_pose_aa"].reshape(1, 63)),
        torch.tensor([[0.0, np.pi / 2, 0.0]], dtype=torch.float32),
        torch.zeros(1, 3),
    )
    np.testing.assert_allclose(frame["smpl_joint_pos_root"],
                               reference["smpl_joints_local"].numpy()[0], atol=1e-6)
    np.testing.assert_allclose(frame["smpl_root_quat_w"], [1.0, 0.0, 0.0, 0.0], atol=2e-7)
    assert np.linalg.norm(frame["smpl_joint_pos_root"][0]) > 0.3
    # Wrists hang below shoulders in the constructed neutral pose.
    assert frame["smpl_joint_pos_root"][20, 2] < frame["smpl_joint_pos_root"][16, 2] - 0.4
    assert frame["smpl_joint_pos_root"][21, 2] < frame["smpl_joint_pos_root"][17, 2] - 0.4


def test_initial_pause_emits_valid_smpl_and_keeps_last_pose(human_asset):
    pytest.importorskip("general_motion_retargeting")
    from sim2real.config.robots import get_robot_cfg
    from sim2real.teleop.pico_retarget_pub import LiveRetargetPublisher

    packets = []
    publisher = LiveRetargetPublisher.__new__(LiveRetargetPublisher)
    publisher.args = SimpleNamespace(
        smpl_human_joints_info_path=str(human_asset), smpl_wire_format="json",
        smpl_topic="pose", smpl_protocol_version=3,
    )
    publisher.robot_cfg = get_robot_cfg("g1")
    publisher._smpl_sock = SimpleNamespace(send_json=lambda payload, flags: packets.append(payload))
    publisher._last_smpl_frame = None
    publisher._smpl_frame_index = 0
    publisher._latest_controller_t_ns = 1234
    publisher.paused_joint_pos = np.zeros(29, dtype=np.float32)
    publisher._publish_paused_smpl_frame()
    assert len(packets) == 1
    expected = build_neutral_smpl_frame(human_asset)
    np.testing.assert_array_equal(packets[0]["smpl_joint_pos_root"][0], expected["smpl_joint_pos_root"])
    publisher._last_smpl_frame["smpl_body_pose_aa"][19, 0] = 0.2
    publisher._publish_paused_smpl_frame()
    assert len(packets) == 2
    np.testing.assert_allclose(packets[1]["smpl_body_pose_aa"][0][19], [0.2, 0.0, 0.0])
    wrist_idx = list(publisher.robot_cfg.joint_names).index("left_wrist_roll_joint")
    assert packets[1]["joint_pos"][0][wrist_idx] == pytest.approx(0.2)
    assert packets[1]["frame_index"] == [1]


def test_official_smpl_replay_preserves_named_robot_joint_order(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sim2real.teleop.sonic_smpl_pkl_pub import load_official_walk

    # Distinct per-joint values detect an accidental Isaac reordering while
    # the outgoing payload continues declaring MuJoCo joint names.
    positions = np.tile(np.arange(29, dtype=np.float32), (3, 1))
    smpl_path, robot_path = tmp_path / "smpl.pkl", tmp_path / "robot.pkl"
    joblib.dump({"pose_aa": np.zeros((3, 72), dtype=np.float32),
                 "smpl_joints": np.zeros((3, 24, 3), dtype=np.float32), "fps": 50.0}, smpl_path)
    joblib.dump({"motion": {"dof": positions, "fps": 50.0}}, robot_path)
    loaded = load_official_walk(smpl_path, robot_path)
    np.testing.assert_array_equal(loaded["joint_pos"], positions)
