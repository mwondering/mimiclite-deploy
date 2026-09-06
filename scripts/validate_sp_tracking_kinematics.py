#!/usr/bin/env python3
"""Compare deploy key-body observations against SP_Tracking's analytic FK API."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from sim2real.rl_policy.observations.sp_tracking import (
    SPV5_2_JOINT_NAMES,
    _DEFAULT_KEYPOINT_SPECS,
    _DEFAULT_KINEMATICS_PATH,
    _SPV52RobotKinematics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    source_root = args.source_repo.resolve() / "src/sp_tracking"
    helper_path = source_root / "tasks/tracking/mdp/motion_fk.py"
    spec = importlib.util.spec_from_file_location("sp_tracking_source_motion_fk", helper_path)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)

    deploy = _SPV52RobotKinematics(
        _DEFAULT_KINEMATICS_PATH, SPV5_2_JOINT_NAMES, _DEFAULT_KEYPOINT_SPECS
    )
    helper = source.MotionFKHelper.from_mjcf_path(
        xml_path=source_root / "assets/robots/g1_tracking_bfm/g1.xml",
        dataset_joint_names=SPV5_2_JOINT_NAMES,
        output_body_names=deploy.physical_names,
        base_body_name="pelvis",
        device="cpu",
    )
    rng = np.random.default_rng(args.seed)
    q = rng.uniform(-0.5, 0.5, (args.trials, 29)).astype(np.float32)
    dq = rng.uniform(-3.0, 3.0, q.shape).astype(np.float32)
    gyro = rng.uniform(-2.0, 2.0, (args.trials, 3)).astype(np.float32)
    with torch.inference_mode():
        pos, quat, lin, ang = helper.body_kinematics(
            torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(gyro)
        )
        parent = deploy.parent_indices
        correction = deploy.correction_indices
        offset = source.quat_apply(quat[:, parent], torch.from_numpy(deploy.local_pos))
        correction_offset = source.quat_apply(
            quat[:, correction], torch.from_numpy(deploy.correction_local_pos)
        )
        key_pos = pos[:, parent] + offset + correction_offset
        key_quat = source.quat_mul(quat[:, parent], torch.from_numpy(deploy.local_quat))
        # Rotating the x/y basis vectors gives the first two matrix columns.
        basis_x = torch.tensor([1.0, 0.0, 0.0])
        basis_y = torch.tensor([0.0, 1.0, 0.0])
        rot6d = torch.cat(
            (source.quat_apply(key_quat, basis_x), source.quat_apply(key_quat, basis_y)),
            dim=-1,
        )
        key_lin = (
            lin[:, parent]
            + torch.cross(ang[:, parent], offset, dim=-1)
            + torch.cross(ang[:, correction], correction_offset, dim=-1)
        )
        key_ang = ang[:, parent] - torch.from_numpy(gyro)[:, None]
        expected = torch.cat(
            [value.flatten(1) for value in (key_pos, rot6d, key_lin, key_ang)], dim=-1
        ).numpy()
    actual = np.concatenate([deploy.compute(qi, dqi, wi) for qi, dqi, wi in zip(q, dq, gyro)])
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    print(json.dumps({
        "source_helper": str(helper_path),
        "trials": args.trials,
        "seed": args.seed,
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "mean_abs_error": float(np.mean(np.abs(actual - expected))),
    }, indent=2))


if __name__ == "__main__":
    main()
