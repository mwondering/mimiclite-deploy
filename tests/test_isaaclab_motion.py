from dataclasses import replace
import mujoco
import numpy as np
import pytest

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.utils.isaaclab_motion import (
    ISAACLAB_G1_JOINT_NAMES,
    prepare_isaaclab_motion,
)
from sim2real.rl_policy.utils.motion import MotionDataset, motion_dataset_first_motion


@pytest.fixture
def clip(tmp_path):
    cfg = get_robot_cfg("g1")
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
    children = "".join(
        f'<body name="link_{i}">{inertial}<joint name="{name}"/></body>'
        for i, name in enumerate(cfg.joint_names)
    )
    xml = tmp_path / "robot.xml"
    xml.write_text(
        f'<mujoco><worldbody><body name="pelvis"><freejoint/>{inertial}'
        f'{children}</body></worldbody></mujoco>'
    )
    cfg = replace(cfg, mjcf_path=str(xml))
    q = np.arange(29, dtype=np.float32)[None, :] * 0.01 + np.arange(3)[:, None] * 0.02
    pos = np.zeros((3, 1, 3), dtype=np.float32)
    pos[:, 0, 0] = [0, 0.04, 0.08]
    pos[:, 0, 2] = 0.8
    quat = np.zeros((3, 1, 4), dtype=np.float32)
    quat[:, :, 0] = 2  # Normalization is required.
    fields = dict(fps=np.array([25]), joint_pos=q, body_pos_w=pos, body_quat_w=quat)
    path = tmp_path / "dance.npz"
    np.savez(path, **fields)
    return cfg, path, fields


def test_isaaclab_joint_mapping_root_and_cache(clip, tmp_path):
    cfg, path, fields = clip
    before = path.read_bytes()
    result = prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None)
    with np.load(result) as data:
        qpos = data["qpos"]
    model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
    for i, name in enumerate(ISAACLAB_G1_JOINT_NAMES):
        np.testing.assert_allclose(qpos[:, model.jnt_qposadr[model.joint(name).id]], fields["joint_pos"][:, i])
    np.testing.assert_allclose(qpos[:, :3], fields["body_pos_w"][:, 0])
    np.testing.assert_allclose(qpos[:, 3:7], np.tile([1, 0, 0, 0], (3, 1)))
    assert path.read_bytes() == before
    assert prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None) == result
    # A canonical manifest clip must bypass adaptation.
    assert prepare_isaaclab_motion(result, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None) is None


def test_both_formats_resample_and_slice(clip, tmp_path):
    cfg, path, _ = clip
    canonical = prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None)
    standalone = MotionDataset.create_from_path(str(path), cfg, target_fps=50)
    original = MotionDataset.create_from_path(canonical, cfg, target_fps=50)
    assert standalone.num_steps > 3
    assert standalone.num_steps == original.num_steps
    selection = motion_dataset_first_motion(standalone)
    args = (np.array([0]), np.array([0]), np.array([-42, 0, 1, 2, 100]))
    actual, expected = selection.get_slice(*args), original.get_slice(*args)
    for field in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))
    # 25 -> 50 Hz must interpolate the midpoint, not just repeat source frames.
    # any4hdmi stores the FK backing in FP16.
    np.testing.assert_allclose(actual.joint_pos[0, 2], (actual.joint_pos[0, 1] + actual.joint_pos[0, 3]) / 2, atol=2e-4)


def test_embedded_names_and_pelvis_index(clip, tmp_path):
    cfg, path, fields = clip
    fields["joint_names"] = np.array(ISAACLAB_G1_JOINT_NAMES[::-1], dtype="S")
    fields["joint_pos"] = fields["joint_pos"][:, ::-1]
    fields["body_names"] = np.array(["other", "pelvis"])
    fields["body_pos_w"] = np.concatenate([fields["body_pos_w"] + 10, fields["body_pos_w"]], axis=1)
    fields["body_quat_w"] = np.repeat(fields["body_quat_w"], 2, axis=1)
    np.savez(path, **fields)
    result = prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None)
    with np.load(result) as data:
        np.testing.assert_allclose(data["qpos"][:, :3], fields["body_pos_w"][:, 1])
        np.testing.assert_allclose(data["qpos"][:, 7], [0, 0.02, 0.04])


@pytest.mark.parametrize("invalid", ["fps", "joint_count", "quaternion", "nan"])
def test_invalid_motion_is_rejected(clip, tmp_path, invalid):
    cfg, path, fields = clip
    if invalid == "fps":
        fields["fps"] = np.array([0])
    elif invalid == "joint_count":
        fields["joint_pos"] = fields["joint_pos"][:, :-1]
    elif invalid == "quaternion":
        fields["body_quat_w"][:] = 0
    else:
        fields["joint_pos"][0, 0] = np.nan
    np.savez(path, **fields)
    with pytest.raises(ValueError):
        prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None)


def test_other_inputs_bypass_conversion(clip, tmp_path):
    cfg, _, _ = clip
    for path in ["hf://example/data/motions/walk.npz", str(tmp_path)]:
        assert prepare_isaaclab_motion(path, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None) is None
    legacy = tmp_path / "motion.npz"
    np.savez(legacy, joint_pos=np.zeros((1, 29)), body_pos_w=np.zeros((1, 1, 3)), body_quat_w=np.zeros((1, 1, 4)))
    (tmp_path / "meta.json").write_text("{}")
    assert prepare_isaaclab_motion(legacy, robot_cfg=cfg, base_dir=tmp_path, mjcf_path=None) is None
