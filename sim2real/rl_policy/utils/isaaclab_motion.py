"""Adapt standalone IsaacLab/SP-Tracking G1 clips to the any4hdmi loader."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model, resolve_mjcf_path
from loguru import logger

from sim2real.config.robots.base import RobotCfg


# SP-Tracking's unnamed IsaacLab exports use articulation (breadth-first) order.
# This differs from the MuJoCo/Unitree joint order used by the policy runtime.
ISAACLAB_G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def _names(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values.reshape(-1)]


def prepare_isaaclab_motion(
    root_path: str | Path,
    *,
    robot_cfg: RobotCfg,
    base_dir: Path,
    mjcf_path: str | Path | None,
) -> str | None:
    """Return a cached qpos clip, or None to preserve the existing loader path.

    Only standalone local NPZs with the IsaacLab body/joint fields are adapted.
    Existing manifest datasets and legacy motion.npz + meta.json stay unchanged.
    Body zero is pelvis in unnamed G1 exports; named exports resolve it by name.
    FK and velocities are recomputed by any4hdmi from root pose and joint angles.
    """
    path = Path(root_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if path.suffix != ".npz" or not path.is_file():
        return None
    if any((p / "manifest.json").is_file() for p in path.parents):
        return None
    if path.name == "motion.npz" and (path.parent / "meta.json").is_file():
        return None

    with np.load(path, allow_pickle=False) as data:
        if not {"joint_pos", "body_pos_w", "body_quat_w"}.issubset(data.files):
            return None
        if robot_cfg.name != "g1":
            raise ValueError("Standalone IsaacLab motion support currently requires robot=g1")
        joints = np.asarray(data["joint_pos"], dtype=np.float32)
        positions = np.asarray(data["body_pos_w"], dtype=np.float32)
        quats = np.asarray(data["body_quat_w"], dtype=np.float32)
        if "fps" not in data or data["fps"].size != 1:
            raise ValueError(f"{path}: IsaacLab motion requires a scalar fps")
        fps = float(data["fps"].reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: fps must be finite and positive")
        names = _names(data["joint_names"]) if "joint_names" in data else list(ISAACLAB_G1_JOINT_NAMES)
        if len(set(names)) != len(names) or set(names) != set(robot_cfg.joint_names):
            raise ValueError(f"{path}: joint_names must contain each G1 joint exactly once")
        if joints.ndim != 2 or joints.shape[0] == 0 or joints.shape[1] != len(names):
            raise ValueError(f"{path}: expected nonempty joint_pos [T, {len(names)}], got {joints.shape}")
        if positions.ndim != 3 or positions.shape[0] != len(joints) or positions.shape[1] == 0 or positions.shape[2] != 3:
            raise ValueError(f"{path}: expected body_pos_w [T, B, 3], got {positions.shape}")
        if quats.shape != (*positions.shape[:2], 4):
            raise ValueError(f"{path}: body_quat_w must match body_pos_w with four wxyz components")
        root_index = 0
        if "body_names" in data:
            body_names = _names(data["body_names"])
            if len(body_names) != positions.shape[1] or body_names.count("pelvis") != 1:
                raise ValueError(f"{path}: body_names must match body arrays and contain one pelvis")
            root_index = body_names.index("pelvis")
        root_pos = positions[:, root_index]
        root_quat = quats[:, root_index]
        if not all(np.isfinite(x).all() for x in (joints, root_pos, root_quat)):
            raise ValueError(f"{path}: motion contains non-finite joint or root values")
        norms = np.linalg.norm(root_quat, axis=-1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError(f"{path}: root quaternion has zero norm")
        root_quat = root_quat / norms

    model_path = (
        resolve_mjcf_path(mjcf_path, dataset_root=base_dir)
        if mjcf_path is not None else robot_cfg.resolve_mjcf_path()
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1 or model.nq != len(names) + 7:
        raise ValueError("IsaacLab G1 conversion requires one free root and 29 scalar joints")
    qpos = np.broadcast_to(model.qpos0, (len(joints), model.nq)).copy()
    root_adr = int(model.jnt_qposadr[free_joints[0]])
    qpos[:, root_adr:root_adr + 3] = root_pos
    qpos[:, root_adr + 3:root_adr + 7] = root_quat
    for source_idx, name in enumerate(names):
        joint = model.joint(name)
        if model.jnt_type[joint.id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Expected scalar joint {name}")
        qpos[:, int(model.jnt_qposadr[joint.id])] = joints[:, source_idx]

    model_reference = mjcf_path if mjcf_path is not None else robot_cfg.mjcf_path
    digest = hashlib.sha256(b"isaaclab-qpos-v3")
    digest.update(qpos.tobytes())
    digest.update(f"{fps}:{model_path}:{model_reference}".encode())
    digest.update(Path(model_path).read_bytes())
    cache_parent = base_dir / ".cache" / "motion" / "isaaclab"
    cache_root = cache_parent / digest.hexdigest()[:24]
    motion_path = cache_root / "motions" / "motion.npz"
    if not motion_path.is_file():
        cache_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_parent) as tmp:
            tmp_root = Path(tmp)
            save_motion(tmp_root / "motions" / "motion.npz", qpos)
            write_manifest(
                # Preserve HF URIs: any4hdmi resolves local XML symlinks to
                # blobs, which would lose the relative mesh directory.
                tmp_root, dataset_name=path.stem,
                mjcf=model_reference if str(model_reference).startswith("hf://") else str(Path(model_path).absolute()),
                timestep=1.0 / fps, qpos_names=qpos_names_from_model(model),
                num_motions=1, total_hours=len(joints) / fps / 3600,
                source={"format": "isaaclab", "path": str(path.resolve())},
            )
            # Publish complete files together. Another loader may win the race.
            try:
                tmp_root.rename(cache_root)
            except OSError:
                if not motion_path.is_file():
                    raise
    logger.info("Loading IsaacLab motion {} via {}; recomputing FK and velocities", path, motion_path)
    return str(motion_path)
