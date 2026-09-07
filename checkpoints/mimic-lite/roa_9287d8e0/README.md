# Official MimicLite-ROA / 9287d8e0

Downloaded on 2026-09-07 from the deploy artifact folder linked by the
[official MimicLite release page](https://github.com/Roboparty/MimicLite/blob/3963976de8778d9292305fc8efacbcae79ed6685/README.md).
The official model is a 16 × 16384 G1 mixture Huge PPO-ROA policy
(`train -> adapt -> finetune`), with actor hidden dimensions `[1024, 1024, 1024]`.

- [Official deployment files](https://drive.google.com/drive/folders/1AFcvP4oDbEskx-wip5bJN-JBaUvwp8MH)
- [Official training run](https://wandb.ai/elijahgalahad/mimic_lite/runs/9287d8e0)
- ONNX Drive file ID: `1xtHn7tu2s-A8RQ84AuT1EqiXQBYRe9pt`.
- YAML Drive file ID: `1z5qIRs78-k6syKcTEoiEvi-Jp3VBhaYy`.

Both files were downloaded using `rclone copyurl` and are unmodified.
These locally calculated SHA-256 values record download identity, not an
independently published upstream checksum:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `policy.onnx` | 26,853,863 | `78aec8b2738f3940fb47a58021b9584faa3f69b53476551c52a2026909db33c1` |
| `policy.yaml` | 6,756 | `47be255e107d0447832b4819b555fe6c97208687b952f58b84338e6aeabf5069` |

## Run and compatibility

Use the root project with `inference-cpu`. Prepare the local motion and G1 asset
cache first. Run from the repository root:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
uv run --no-sync python sim2real/sim_env/integrated_sim2sim.py \
  --policy-config checkpoints/mimic-lite/roa_9287d8e0/policy.yaml \
  --motion-path datasets/lafan40/motions/walk1_subject1.npz \
  --inference-backend onnx-cpu --env-dt 0.02 --initial-pause-s 2
```

The existing runtime consumes `command[304]` and `policy[535]`; the ONNX emits
`action[29]` and `priv_student[1024]`. The future-reference horizon is four
50 Hz frames (0.08 s). The official graph uses IR 10 / opset 20. Older onboard
ONNX Runtime builds that do not support this format need a separately validated
conversion or runtime upgrade; this download does not include such a conversion.

## Local validation

ONNX checker, YAML input-name matching, and CPU single-step inference passed.
The integrated MuJoCo smoke test used seed `20260728`, initial pause 2 s, and
`datasets/root90/motions/backward__00__FairySteps.npz` (401 frames).
It ran for 15 simulated seconds, including final-frame holding, and recorded
`outputs/mimiclite_roa_9287d8e0/download_smoke_root.npz`.
Final relative root XY error was 0.477 m. This short test does not establish
general tracking quality or real-robot safety. No physical G1 was controlled.

This directory is included in the deployment repository alongside the two
SP-Tracking checkpoints. No source training checkout or training `.pt` is
required to load these deployment files.
