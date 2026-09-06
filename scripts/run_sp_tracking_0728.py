#!/usr/bin/env python3
"""Run the verified 0728 SPV5-2 checkpoint in simulation or on G1.

Usage: uv run --no-sync python scripts/run_sp_tracking_0728.py MODE [OPTIONS]
Modes: sim2sim, sim2real. Use MODE --help for the runtime's complete options.
Add --dry-run to print the resolved settings without opening robot I/O.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_CONFIG = (
    REPO_ROOT / "checkpoints/sp-tracking/0728_baoshou_waist_dataclean_changedr/policy.yaml"
)
MOTION_PATH = REPO_ROOT / "datasets/lafan40/motions/walk1_subject1.npz"
SOURCE_SHA256 = "65283dfa5f48c51dc28d2873852ecf379b04c2d0a97a23a4bb5b924a486684f2"
ADAPTED_SHA256 = "c88d39cfe823c801dd7ada9d029360744dc692af900ce955824d6904cb978ef4"


def validate_checkpoint(config_path: str) -> None:
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. See the SP-Tracking 0728 deployment tutorial "
            "for the artifact preparation command."
        )
    config = yaml.safe_load(config_path.read_text())
    model_path = config_path.parent / config["model_path"]
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    source = metadata["adaptation"]["source_files"]["policy_22000.onnx"]
    if metadata["iteration"] != 22000 or source["sha256"] != SOURCE_SHA256:
        raise ValueError("The selected artifact is not the requested 0728 / 22000 checkpoint")
    digest = hashlib.sha256()
    with model_path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ADAPTED_SHA256:
        raise ValueError(f"The adapted ONNX checksum does not match: {model_path}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    mode, *arguments = sys.argv[1:]
    if mode not in {"sim2sim", "sim2real"}:
        raise SystemExit(f"Unknown mode {mode!r}; choose sim2sim or sim2real")
    dry_run = "--dry-run" in arguments
    arguments = [argument for argument in arguments if argument != "--dry-run"]

    # Set these before importing any runtime module, including for noninteractive SSH.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")
    import tyro

    if mode == "sim2sim":
        from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs

        args = tyro.cli(
            IntegratedSim2SimArgs,
            default=IntegratedSim2SimArgs(
                policy_config=str(POLICY_CONFIG),
                motion_path=str(MOTION_PATH),
                initial_pause_s=2.0,
                seed=20260728,
            ),
            args=arguments,
        )
        if args.env_dt != 0.02:
            raise ValueError("This checkpoint requires a 50 Hz policy loop (--env-dt 0.02)")
        run = lambda: IntegratedSim2Sim(args).run()
    else:
        from sim2real.rl_policy.tracking import Tracking, TrackingArgs

        args = tyro.cli(
            TrackingArgs,
            default=TrackingArgs(
                policy_config=str(POLICY_CONFIG),
                robot_io="inline",
                controller="joystick",
                motion_backend="npz",
                motion_path=str(MOTION_PATH),
            ),
            args=arguments,
        )
        if args.rl_rate != 50.0:
            raise ValueError("This checkpoint requires a 50 Hz policy loop (--rl-rate 50)")
        run = lambda: Tracking(args=args).run()

    validate_checkpoint(args.policy_config)
    if dry_run:
        print(json.dumps({"mode": mode, **asdict(args)}, indent=2))
        return
    run()


if __name__ == "__main__":
    main()
