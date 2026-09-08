#!/usr/bin/env python3
"""Build SONIC release YAML from the upstream C++ policy constants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2real.config.robots import get_robot_cfg


def source_parameters(source_repo: Path) -> tuple[dict, Path]:
    header = source_repo / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
    # Compile the source expressions, including float rounding of kp/kd, rather
    # than copying another policy's gains or evaluating C++ text as Python.
    program = r'''
#include <vector>
#include <iostream>
#include <iomanip>
#include "policy_parameters.hpp"
template<class T> void emit(const char* key, const T& values, bool last=false) {
  std::cout << "\"" << key << "\":[";
  for (size_t i=0; i<values.size(); ++i) { if(i) std::cout << ","; std::cout << values[i]; }
  std::cout << "]" << (last ? "" : ",");
}
int main() {
  std::cout << std::setprecision(17) << "{";
  emit("mujoco_to_isaaclab", mujoco_to_isaaclab);
  emit("default_angles", default_angles);
  emit("action_scale", g1_action_scale);
  emit("kp", kps); emit("kd", kds, true);
  std::cout << "}\n";
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        cpp, executable = Path(tmp) / 'parameters.cpp', Path(tmp) / 'parameters'
        cpp.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-I', str(header.parent), str(cpp), '-o', str(executable)], check=True)
        values = json.loads(subprocess.check_output([str(executable)], text=True))
    return values, header


def make_config(mode: str, values: dict) -> dict:
    cfg = get_robot_cfg('g1')
    joints = list(cfg.joint_names)
    policy_joints = [joints[i] for i in values['mujoco_to_isaaclab']]
    future_steps = list(range(0, 50, 5)) if mode == 'g1' else list(range(10))
    def obs(target, **kwargs):
        return {'_target_': f'sonic.{target}', **kwargs}
    if mode == 'g1':
        reference = {
            'command': obs('sonic_command_multi_future_nonflat', future_steps=future_steps),
            'root_orientation': obs('sonic_motion_anchor_ori_b_mf_nonflat', future_steps=future_steps, heading_method='source_cpp'),
        }
    else:
        reference = {
            'smpl_joints': obs('sonic_smpl_joints_multi_future_local', future_steps=future_steps),
            'root_orientation': obs('sonic_smpl_root_ori_b_multi_future', future_steps=future_steps, heading_method='source_cpp'),
            'wrists': obs('sonic_joint_pos_multi_future_wrist_for_smpl', future_steps=future_steps),
        }
    # The source records its first sensor frame only after entering CONTROL.
    # Preserve its missing-history padding through the explicit release option.
    history = {'history_steps': list(range(10)), 'history_initialization': 'source_cpp'}
    proprioception = {
        'base_ang_vel': obs('sonic_root_ang_vel_history', **history),
        'joint_pos': obs('sonic_joint_pos_rel_history', joint_order='policy', **history),
        'joint_vel': obs('sonic_joint_vel_history', joint_order='policy', **history),
        'actions': obs('sonic_prev_actions_history', **history),
        'gravity_dir': obs('sonic_projected_gravity_history', **history),
    }
    return {
        'model_path': 'policy.onnx',
        'observation': {f'{mode}_input': reference, 'proprioception': proprioception},
        'joint_names_simulation': joints,
        'body_names_simulation': list(cfg.body_names),
        'policy_joint_names': policy_joints,
        'default_joint_pos': dict(zip(joints, values['default_angles'])),
        'joint_kp': dict(zip(joints, values['kp'])),
        'joint_kd': dict(zip(joints, values['kd'])),
        'action_scale': dict(zip(joints, values['action_scale'])),
        'motion': {
            'motion_backend': 'zmq' if mode == 'g1' else 'smpl_zmq',
            'motion_zmq_connect': f'tcp://127.0.0.1:{28701 if mode == "g1" else 28702}',
            'motion_dt_s': 0.02, 'motion_tolerance_s': 0.04,
            'future_steps': future_steps, 'joint_names': policy_joints,
            'body_names': ['pelvis'], 'root_body_name': 'pelvis',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'checkpoints/sonic/release')
    args = parser.parse_args()
    values, header = source_parameters(args.source_repo.resolve())
    for mode in ('g1', 'smpl'):
        folder = args.output_dir / mode
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'policy.yaml').write_text(yaml.safe_dump(make_config(mode, values), sort_keys=False))
    contract = {
        'source_header': str(header.relative_to(args.source_repo.resolve())),
        'source_header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
        'parameters': values,
        'action_clipping': 'none in source CreatePolicyCommand',
        'history_initialization': 'source_cpp: zeros, with gravity from the padded zero quaternion (+Z)',
    }
    (args.output_dir / 'source_config.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(f'Wrote G1 and SMPL policy YAML under {args.output_dir}')


if __name__ == '__main__':
    main()
