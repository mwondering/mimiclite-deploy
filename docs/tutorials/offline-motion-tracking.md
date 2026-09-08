---
title: Offline Motion Tracking
sidebar_position: 1
---

This tutorial uses the root project tracking policy with an offline motion clip.

Default motion:

```text
hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

## Sim2Sim

### Supported local NPZ formats

`--motion-path` accepts both existing any4hdmi datasets (including HF references)
and standalone IsaacLab/SP-Tracking G1 NPZ clips with `fps`, `joint_pos`,
`body_pos_w`, and `body_quat_w` fields. No renaming or manual conversion is needed.
For unnamed IsaacLab exports, the loader uses the standard 29-joint G1 IsaacLab
order and body index 0 as pelvis; embedded `joint_names` and `body_names` take
precedence. Quaternions must use wxyz order.

Standalone clips are converted into `.cache/motion/isaaclab/` without modifying
the source. The loader preserves the root trajectory and joint angles, then
recomputes body poses and velocities using the robot MJCF (or `motion.mjcf_path`
override) and resamples through any4hdmi to 50 Hz. Original non-root body arrays
and velocity arrays are not copied. This also works in integrated sim2sim.

Start the MuJoCo execution process. It prints the mjviser URL after startup:

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

In a second terminal, start the tracking policy:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

Process roles:

- `sim2real/sim_env/base_sim.py` executes `low_cmd` in MuJoCo and publishes `low_state`.
- `sim2real/rl_policy/tracking.py` consumes `low_state`, runs the exported policy, and publishes the next `low_cmd`.

After both processes are up, press `]` in the policy terminal to start. Use the Elastic Band controls in the mjviser UI to disable or tune the virtual gantry.

## Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). For example, the tracking policy still starts with:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Integrated Sim2Sim

Use the integrated runner when the policy and MuJoCo should live in one process. It loads the policy immediately, sets the robot to the first frame of the motion, waits five seconds, tracks until the motion ends, and then holds the last frame. Elastic band is disabled by default for this runner, and the runner prints its mjviser URL after startup.

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy-config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion-path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

Add `--headless` for non-visual runs. The mjviser scene updates once per environment step when a browser client is connected. In mjviser mode, use the `Restart motion` button after the final-frame hold to reset the robot to the first frame and repeat the wait-track-hold sequence.

For quantitative evaluation, add `--trajectory-output <path>.npz` and compute
motion progress, global root tracking, and local body tracking with the scripts
under `scripts/tracking_experiment/`.

## Next Steps

- [Pico Teleoperation](/tutorials/pico-teleoperation)
