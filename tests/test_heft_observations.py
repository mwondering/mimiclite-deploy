from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim2real.rl_policy.observations.heft import heft_component_obs, heft_policy_obs


REFERENCE_STEPS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
HISTORY_STEPS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
JOINT_NAMES = tuple(f"joint_{idx}" for idx in range(29))
SOURCE_DIR = (Path(__file__).resolve().parents[1] / "external" /
              "motion_tracking_sim2real" / "sim2real" / "src" / "runtime")


def _kwargs() -> dict:
    return dict(
        future_steps=REFERENCE_STEPS,
        root_angvel_history_steps=HISTORY_STEPS,
        projected_gravity_history_steps=HISTORY_STEPS,
        joint_pos_history_steps=HISTORY_STEPS,
        joint_vel_history_steps=HISTORY_STEPS,
        prev_action_steps=8,
        joint_names=JOINT_NAMES,
        root_body_name="pelvis",
        boot_indicator_max=25,
    )


def _environment() -> SimpleNamespace:
    n_ref = len(REFERENCE_STEPS)
    quat = np.zeros((1, n_ref, 1, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    return SimpleNamespace(
        state_processor=SimpleNamespace(
            # Deliberately unlike both source motion and policy order.
            joint_names=list(JOINT_NAMES)[::-1],
            joint_pos=np.zeros(29, dtype=np.float32),
            joint_vel=np.zeros(29, dtype=np.float32),
            root_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            root_ang_vel_b=np.zeros(3, dtype=np.float32),
        ),
        policy_joint_names=list(JOINT_NAMES),
        joint_names_simulation=list(JOINT_NAMES)[::-1],
        body_names_simulation=["pelvis"],
        motion_future_steps=np.asarray(REFERENCE_STEPS),
        motion_joint_names=list(JOINT_NAMES),
        motion_body_names=["pelvis"],
        motion_data=SimpleNamespace(
            joint_pos=np.zeros((1, n_ref, 29), dtype=np.float32),
            joint_vel=np.zeros((1, n_ref, 29), dtype=np.float32),
            body_pos_w=np.zeros((1, n_ref, 1, 3), dtype=np.float32),
            body_quat_w=quat,
            body_lin_vel_w=np.zeros((1, n_ref, 1, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((1, n_ref, 1, 3), dtype=np.float32),
        ),
        total_inference_cnt=0,
    )


@pytest.fixture
def upstream(monkeypatch):
    """Load the actual source helper code, without importing its robot process."""
    if not (SOURCE_DIR / "observation.py").is_file():
        pytest.skip("Source parity requires extracted motion_tracking origin/sim2real checkout")
    monkeypatch.setitem(sys.modules, "runtime", ModuleType("runtime"))
    modules = {}
    for name in ("math_utils", "observation"):
        qualified_name = f"runtime.{name}"
        spec = importlib.util.spec_from_file_location(qualified_name, SOURCE_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified_name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["observation"]


def _source_modules(upstream, controller, policy):
    # Exact TrackingPolicyRaw._build_obs_modules() order, PMG variant.
    return [
        upstream.BootIndicator(policy),
        upstream.TrackingCommandObsRaw(controller, policy),
        upstream.TargetJointPosObs(policy),
        upstream.TargetRootZObs(policy),
        upstream.TargetProjectedGravityBObs(policy),
        upstream.RootAngVelBHistory(controller, policy),
        upstream.ProjectedGravityBHistory(controller, policy),
        upstream.JointPos(controller, policy),
        upstream.JointVel(controller, policy),
        upstream.PrevActions(policy),
    ]


@pytest.mark.parametrize("reference_index", [0, 20, 38])
@pytest.mark.parametrize("quaternion_scale", [1.0, 1.03])
def test_pmg_matches_actual_source_observations_across_history_and_reset(
    upstream, reference_index, quaternion_scale
):
    rng = np.random.default_rng(1729)
    env = _environment()
    core = heft_policy_obs(env=env, **_kwargs())
    controller = SimpleNamespace(config=SimpleNamespace(policy_joint_names=JOINT_NAMES))
    policy = SimpleNamespace(
        config=SimpleNamespace(**_kwargs()), controller=controller, n_joints=29,
        last_action=np.zeros(29, dtype=np.float32), ref_len=39, ref_idx=reference_index,
        ref_joint_pos=rng.normal(size=(39, 29)).astype(np.float32),
        ref_root_pos=rng.normal(size=(39, 3)).astype(np.float32),
        ref_root_quat=(Rotation.random(39, random_state=rng).as_quat(scalar_first=True)
                       * quaternion_scale).astype(np.float32),
    )
    modules = _source_modules(upstream, controller, policy)
    selected = np.clip(reference_index + np.asarray(REFERENCE_STEPS), 0, 38)
    # The receiver selects exact signed steps; history clamps at clip edges.
    env.motion_data.joint_pos[0] = policy.ref_joint_pos[selected]
    env.motion_data.body_pos_w[0, :, 0] = policy.ref_root_pos[selected]
    env.motion_data.body_quat_w[0, :, 0] = policy.ref_root_quat[selected]
    for module in modules:
        module.reset()
    core.reset()
    for step in range(53):
        env.total_inference_cnt = step
        controller.qj = rng.normal(size=29).astype(np.float32)
        controller.dqj = rng.normal(size=29).astype(np.float32)
        controller.gyro = rng.normal(size=3).astype(np.float32)
        controller.quat = (Rotation.random(random_state=rng).as_quat(scalar_first=True)
                           * quaternion_scale).astype(np.float32)
        state = env.state_processor
        state.joint_pos[:] = controller.qj[::-1]
        state.joint_vel[:] = controller.dqj[::-1]
        state.root_ang_vel_b[:] = controller.gyro
        state.root_quat_w[:] = controller.quat
        last_action = rng.uniform(-10.0, 10.0, 29).astype(np.float32)
        policy.last_action[:] = last_action
        if step in (0, 27):
            # Source resets its last output; generic sim2real retains it.
            core.reset()
            for module in modules:
                module.reset()
            policy.last_action[:] = 0.0
        for module in modules:
            module.update()
        expected = np.concatenate([module.compute() for module in modules])
        core.update({"action": last_action})
        np.testing.assert_allclose(core.compute()[0], expected, atol=2e-6, rtol=2e-6)


def test_semantic_groups_share_one_update_and_cached_outputs():
    env = _environment()
    # Reorder the groups: their construction order must not alter semantics.
    names = ("proprioception", "context", "target_motion", "motion_command")
    groups = {name: heft_component_obs(component=name, env=env, **_kwargs()) for name in names}
    assert len({id(group._core) for group in groups.values()}) == 1
    for group in groups.values():
        group.reset()
    env.state_processor.joint_pos[:] = np.arange(29)
    for group in groups.values():
        group.update({"action": np.ones(29, dtype=np.float32)})
    core = groups["context"]._core
    assert core.component_dims == dict(context=1, motion_command=114, target_motion=806,
                                       proprioception=808)
    assert groups["context"].compute().item() == pytest.approx(24 / 25)
    snapshot = {name: group.compute().copy() for name, group in groups.items()}
    env.state_processor.joint_pos[:] = -100.0
    # Repeated group reads or updates during the same inference cannot mutate
    # a partly consumed observation or advance the sparse history again.
    for group in groups.values():
        group.update({"action": np.ones(29, dtype=np.float32)})
    for name, group in groups.items():
        np.testing.assert_array_equal(group.compute(), snapshot[name])
    assert np.count_nonzero(core._joint_pos_history[1:]) == 0
    np.testing.assert_array_equal(core._prev_actions, 0.0)
    env.total_inference_cnt += 1
    for group in groups.values():
        group.update({"action": np.full(29, 2.0, dtype=np.float32)})
    np.testing.assert_array_equal(core._prev_actions[0], 2.0)
    np.testing.assert_array_equal(core._prev_actions[1:], 0.0)
    np.testing.assert_array_equal(core._joint_pos_history[1], np.arange(29)[::-1])


def test_live_motion_body_and_joint_reordering_preserves_reference_semantics():
    env = _environment()
    rng = np.random.default_rng(514)
    env.motion_data.joint_pos[:] = rng.normal(size=env.motion_data.joint_pos.shape)
    env.motion_data.body_pos_w[:] = (0.2, -0.3, 0.8)
    env.motion_data.body_quat_w[:] = (0.5, 0.5, 0.5, 0.5)
    core = heft_policy_obs(env=env, **_kwargs())
    core.update({})
    expected = {name: core._parts[name].copy() for name in ("motion_command", "target_motion")}
    # PICO includes a world body; startup's reference fallback omits it.
    env.motion_body_names = ["world", "pelvis"]
    motion = env.motion_data
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        value = getattr(motion, name)
        world = np.zeros_like(value)
        if name == "body_quat_w":
            world[..., 0] = 1.0
        setattr(motion, name, np.concatenate((world, value), axis=2))
    env.motion_joint_names.reverse()
    motion.joint_pos = motion.joint_pos[:, :, ::-1].copy()
    motion.joint_vel = motion.joint_vel[:, :, ::-1].copy()
    env.total_inference_cnt += 1
    core.update({})
    for name, value in expected.items():
        np.testing.assert_array_equal(core._parts[name], value)
    # The pelvis height cannot silently become the world body's zero height.
    np.testing.assert_array_equal(core.ref_root_pos_future_w[..., 2], np.float32(0.8))


def test_reset_discards_previous_run_actions_and_sparse_history():
    env = _environment()
    core = heft_policy_obs(env=env, **_kwargs())
    for step in range(30):
        env.total_inference_cnt = step
        env.state_processor.joint_pos[:] = step + 1
        core.update({"action": np.full(29, 7.0, dtype=np.float32)})
    core.reset()
    env.state_processor.joint_pos[:] = 100.0
    core.update({"action": np.full(29, 7.0, dtype=np.float32)})
    np.testing.assert_array_equal(core._joint_pos_history[0], 100.0)
    np.testing.assert_array_equal(core._joint_pos_history[1:], 0.0)
    np.testing.assert_array_equal(core._prev_actions, 0.0)
    assert core._parts["context"].item() == pytest.approx(24 / 25)


@pytest.mark.parametrize("steps", [(), (-1,), (0, -2)])
def test_rejects_invalid_history_offsets(steps):
    kwargs = _kwargs()
    kwargs["root_angvel_history_steps"] = steps
    with pytest.raises(ValueError, match="non-empty and non-negative"):
        heft_policy_obs(env=_environment(), **kwargs)
