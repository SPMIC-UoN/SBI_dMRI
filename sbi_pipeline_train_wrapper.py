#!/usr/bin/env python3
"""
sbi_pipeline_train_wrapper.py

Convenience wrapper:
  1) simulate_data.py (creates run_dir + dataset)
  2) train_snpe.py    (trains and saves models into same run_dir)

This is just glue so you can run one command.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser("sbi_pipeline_train.py")
    p.add_argument("configfile", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)

    p.add_argument("--codec", type=str, default="lzma", choices=["lzma", "gzip", "none"])
    p.add_argument("--preset", type=int, default=6)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--density", type=str, default="nsf")

    # x0 storage
    p.add_argument("--no-save-x0", action="store_true", help="Disable storing noisefree x0.")

    args = p.parse_args()

    # 1) simulate
    sim_cmd = [
        "python", "simulate_data.py",
        args.configfile,
        "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--codec", args.codec,
        "--preset", str(args.preset),
    ]
    if args.no_save_x0:
        sim_cmd.append("--no-save-x0")

    subprocess.check_call(sim_cmd)

    # We need the most recent run dir.
    # Convention: results_root/model_name/run_*/...
    # We read results_root + model_name from the config by importing it.
    cfg = {}
    exec(Path(args.configfile).read_text(), cfg, cfg)
    results_root = cfg["paths"]["results_root"]
    model_name = cfg["training_params"]["model_name"]

    model_dir = Path(results_root) / model_name
    run_dirs = sorted([p for p in model_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run directories found under: {model_dir}")
    run_dir = run_dirs[-1]

    # 2) train
    train_cmd = [
        "python", "train_snpe.py",
        "--run-dir", str(run_dir),
        "--codec", args.codec,
        "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--device", args.device,
        "--density", args.density,
    ]
    subprocess.check_call(train_cmd)

    print(f"\n✅ Finished. Run folder:\n{run_dir}\n")


if __name__ == "__main__":
    main()