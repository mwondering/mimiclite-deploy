"""Release SONIC parity against the actual source C++ observation helpers."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim2real.rl_policy.observations.sonic import (
    sonic_command_multi_future_nonflat,
    sonic_joint_pos_multi_future_wrist_for_smpl,
    sonic_joint_pos_rel_history,
    sonic_joint_vel_history,
    sonic_motion_anchor_ori_b_mf_nonflat,
    sonic_prev_actions_history,
    sonic_projected_gravity_history,
    sonic_root_ang_vel_history,
    sonic_smpl_joints_multi_future_local,
    sonic_smpl_root_ori_b_multi_future,
)


SOURCE = Path(os.environ.get(
    "GROOT_SOURCE_DIR",
    str(Path(__file__).resolve().parents[2] / "GR00T-WholeBodyControl"),
)) / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"


def _source_method(text: str, name: str) -> str:
    start = text.index(f"bool {name}(")
    # Parameters have {} default arguments, so locate the actual body after ') {'.
    start_body = text.index(") {", start) + 2
    depth = 1
    end = start_body + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.fixture(scope="module")
def source_cpp(tmp_path_factory):
    if not SOURCE.exists() or shutil.which("g++") is None:
        pytest.skip("GR00T source checkout and g++ are required for source parity")
    cpp_source = (SOURCE / "src/g1_deploy_onnx_ref.cpp").read_text()
    methods = "\n".join(_source_method(cpp_source, name) for name in (
        "GatherHisBaseAngularVelocity", "GatherHisBodyJointPositions",
        "GatherHisBodyJointVelocities", "GatherHisLastActions", "GatherHisGravityDir",
    ))
    tmp = tmp_path_factory.mktemp("sonic_source_cpp")
    wrapper = tmp / "source_parity.cpp"
    wrapper.write_text(r'''
#include <iostream>
#include <memory>
#include <tuple>
#include "state_logger.hpp"
#include "math_utils.hpp"
struct Reference {
  double control_dt_ = 0.02;
  std::unique_ptr<StateLogger> state_logger_ = std::make_unique<StateLogger>("", 64, 29, 29, 0.02, false);
''' + methods + r'''
};
int main(int argc, char** argv) {
  std::cout << std::setprecision(17);
  if (argc > 1) {
    std::array<double, 4> initial_robot, initial_ref, robot, ref;
    while (std::cin >> initial_robot[0]) {
      for (int j=1; j<4; ++j) std::cin >> initial_robot[j];
      for (auto& v : initial_ref) std::cin >> v;
      for (auto& v : robot) std::cin >> v;
      for (auto& v : ref) std::cin >> v;
      auto offset = quat_mul_d(calc_heading_quat_d(initial_robot), calc_heading_quat_inv_d(initial_ref));
      auto rel = quat_mul_d(quat_conjugate_d(robot), quat_mul_d(offset, ref));
      auto mat = quat_to_rotation_matrix_d(rel);
      for (int i=0; i<3; ++i) for (int j=0; j<2; ++j) std::cout << mat[i][j] << ' ';
      std::cout << '\n';
    }
    return 0;
  }
  Reference r;
  std::array<double, 4> quat;
  std::array<double, 3> omega, zero3{};
  std::vector<double> q(29), dq(29), action(29), zero29(29), zero58(58), zero7(7);
  while (std::cin >> quat[0]) {
    for (int j=1; j<4; ++j) std::cin >> quat[j];
    for (auto& v : omega) std::cin >> v;
    for (auto& v : q) std::cin >> v;
    for (auto& v : dq) std::cin >> v;
    for (auto& v : action) std::cin >> v;
    r.state_logger_->LogFullState(quat, omega, zero3, quat, zero3, zero3,
      std::span(q), std::span(dq), std::span(action), std::span(zero58),
      std::span(zero29), std::span(zero29), std::span(zero7), std::span(zero7),
      std::span(zero7), std::span(zero7), std::span(zero7), std::span(zero7), 0.0);
    std::vector<double> obs(930);
    if (!r.GatherHisBaseAngularVelocity(obs, 0, 10, 1)
      || !r.GatherHisBodyJointPositions(obs, 30, 10, 1)
      || !r.GatherHisBodyJointVelocities(obs, 320, 10, 1)
      || !r.GatherHisLastActions(obs, 610, 10, 1)
      || !r.GatherHisGravityDir(obs, 900, 10, 1)) return 2;
    for (auto v : obs) std::cout << v << ' ';
    std::cout << '\n';
  }
}
''')
    binary = tmp / "source_parity"
    subprocess.run([
        "g++", "-std=c++20", "-O0", "-pthread", "-I", str(SOURCE / "include"),
        str(wrapper), str(SOURCE / "src/state_logger.cpp"),
        str(SOURCE / "src/file_sink.cpp"), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    return binary


def _source_output(binary, inputs, *args):
    result = subprocess.run(
        [str(binary), *args], input="\n".join(
            " ".join(format(float(x), ".17g") for x in row) for row in inputs
        ) + "\n", capture_output=True, text=True, check=True,
    )
    return np.asarray([
        np.fromstring(line, sep=" ") for line in result.stdout.splitlines() if line.strip()
    ])


def _env():
    # Policy, simulator/state, and live motion deliberately use different orders.
    names = [f"joint_{i}" for i in range(23)] + [
        "left_wrist_roll_joint", "right_wrist_roll_joint",
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    ]
    state_names = names[::-1]
    state = SimpleNamespace(
        joint_names=state_names, joint_pos=np.zeros(29), joint_vel=np.zeros(29),
        root_ang_vel_b=np.zeros(3), root_quat_w=np.array([1., 0., 0., 0.]),
    )
    future = 10
    return SimpleNamespace(
        state_processor=state, policy_joint_names=names,
        joint_names_simulation=state_names, default_dof_angles=np.linspace(-.3, .6, 29),
        num_actions=29, motion_future_steps=np.arange(future),
        motion_joint_names=names[7:] + names[:7], motion_body_names=["world", "pelvis"],
        body_names_simulation=["world", "pelvis"],
        motion_data=SimpleNamespace(
            joint_pos=np.zeros((1, future, 29)), joint_vel=np.zeros((1, future, 29)),
            body_pos_w=np.zeros((1, future, 2, 3)), body_lin_vel_w=np.zeros((1, future, 2, 3)),
            body_quat_w=np.tile([1., 0., 0., 0.], (1, future, 2, 1)),
            body_ang_vel_w=np.zeros((1, future, 2, 3)),
            smpl_joint_pos_root=np.zeros((1, future, 24, 3)),
            smpl_root_quat_w=np.tile([1., 0., 0., 0.], (1, future, 1)),
        ),
    )


def _history_observations(env, initialization="source_cpp"):
    args = dict(env=env, history_steps=list(range(10)), history_initialization=initialization)
    return [
        sonic_root_ang_vel_history(**args),
        sonic_joint_pos_rel_history(**args, joint_order="policy"),
        sonic_joint_vel_history(**args, joint_order="policy"),
        sonic_prev_actions_history(**args),
        sonic_projected_gravity_history(**args),
    ]


def test_all_930_history_values_match_actual_cpp_gatherers(source_cpp):
    env = _env()
    observations = _history_observations(env)
    rng = np.random.default_rng(20260908)
    source_inputs, actual = [], []
    for tick in range(25):
        q_rel, dq, action = rng.normal(size=(3, 29))
        if tick == 0:
            action[:] = 0
        omega = rng.normal(size=3)
        quat = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
        env.state_processor.root_quat_w = quat
        env.state_processor.root_ang_vel_b = omega
        env.state_processor.joint_pos = q_rel[::-1] + env.default_dof_angles
        env.state_processor.joint_vel = dq[::-1]
        for obs in observations:
            obs.update({"action": action})
        actual.append(np.concatenate([obs.compute() for obs in observations]))
        source_inputs.append(np.r_[quat, omega, q_rel, dq, action])
    expected = _source_output(source_cpp, source_inputs)
    np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=2e-6)
    # This unintuitive initial padding is what the source C++ actually produces.
    np.testing.assert_array_equal(expected[0, 900:927].reshape(9, 3), np.tile([0, 0, 1], (9, 1)))


def test_source_reset_discards_stale_action_and_restarts_history():
    env = _env()
    obs = sonic_prev_actions_history(env=env, history_steps=[0, 1, 2], history_initialization="source_cpp")
    obs.update({"action": np.ones(29)})
    np.testing.assert_array_equal(obs.compute(), 0)
    obs.update({"action": np.ones(29) * 2})
    np.testing.assert_array_equal(obs.compute().reshape(3, 29)[-1], 2)
    obs.reset()
    obs.update({"action": np.ones(29) * 7})
    np.testing.assert_array_equal(obs.compute(), 0)


def test_default_history_preserves_repeat_first_and_simulation_order():
    env = _env()
    obs = sonic_joint_pos_rel_history(env=env, history_steps=[0, 1, 2])
    value = np.arange(29) / 10
    env.state_processor.joint_pos = value + env.default_dof_angles
    obs.update({})
    np.testing.assert_allclose(obs.compute().reshape(3, 29), np.tile(value, (3, 1)), atol=1e-7)


def test_motion_command_is_all_positions_then_all_velocities_in_policy_order():
    env = _env()
    env.motion_data.joint_pos[:] = np.arange(290).reshape(1, 10, 29)
    env.motion_data.joint_vel[:] = 1000 + np.arange(290).reshape(1, 10, 29)
    obs = sonic_command_multi_future_nonflat(env=env, future_steps=list(range(10)))
    obs.update({})
    indices = [env.motion_joint_names.index(name) for name in env.policy_joint_names]
    expected = np.r_[env.motion_data.joint_pos[0][:, indices].ravel(), env.motion_data.joint_vel[0][:, indices].ravel()]
    np.testing.assert_array_equal(obs.compute(), expected)


@pytest.mark.parametrize("kind", ["g1", "smpl"])
def test_orientations_match_actual_cpp_math_with_heading_reset(source_cpp, kind):
    env = _env()
    # A near-vertical initial pose exercises the source-vs-HDMI heading difference.
    init_robot = Rotation.from_euler("xyz", [.5, np.pi / 2 + .03, .7]).as_quat()[[3, 0, 1, 2]]
    init_ref = Rotation.from_euler("xyz", [-.2, -.4, -.8]).as_quat()[[3, 0, 1, 2]]
    env.state_processor.root_quat_w = init_robot
    env.motion_data.body_quat_w[:, :, 1] = init_ref
    env.motion_data.smpl_root_quat_w[:] = init_ref
    cls = sonic_motion_anchor_ori_b_mf_nonflat if kind == "g1" else sonic_smpl_root_ori_b_multi_future
    obs = cls(env=env, future_steps=list(range(10)), heading_method="source_cpp")
    obs.reset()
    rng = np.random.default_rng(310)
    robot = Rotation.from_euler("xyz", [.3, -.2, 1.2]).as_quat()[[3, 0, 1, 2]]
    refs = Rotation.random(10, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    env.state_processor.root_quat_w = robot
    env.motion_data.body_quat_w[0, :, 1] = refs
    env.motion_data.smpl_root_quat_w[0] = refs
    obs.update({})
    expected = _source_output(source_cpp, [np.r_[init_robot, init_ref, robot, q] for q in refs], "orientation")
    np.testing.assert_allclose(obs.compute().reshape(10, 6), expected, atol=3e-6)


def test_smpl_semantic_components_keep_per_frame_joint_and_wrist_order():
    env = _env()
    env.motion_data.smpl_joint_pos_root[:] = np.arange(720).reshape(1, 10, 24, 3)
    env.motion_data.joint_pos[:] = np.arange(290).reshape(1, 10, 29)
    joints = sonic_smpl_joints_multi_future_local(env=env, future_steps=list(range(10)))
    wrists = sonic_joint_pos_multi_future_wrist_for_smpl(env=env, future_steps=list(range(10)))
    np.testing.assert_array_equal(joints.compute(), np.arange(720))
    indices = [env.motion_joint_names.index(name) for name in wrists.WRIST_JOINT_NAMES]
    np.testing.assert_array_equal(wrists.compute(), env.motion_data.joint_pos[0][:, indices].ravel())
    env.motion_joint_names.reverse()
    env.motion_data.joint_pos = env.motion_data.joint_pos[:, :, ::-1].copy()
    np.testing.assert_array_equal(wrists.compute(), np.arange(290).reshape(10, 29)[:, indices].ravel())
