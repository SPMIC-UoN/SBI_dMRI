#!/usr/bin/env python3
"""
train_snpe.py

Standalone SNPE-C training from pre-simulated (theta, x) tensors.

Why this exists:
- You can run simulation and training independently.
- You can store huge datasets in float16 for space, but train in float32.

Input shapes supported:
- theta: (N,P) or (...,P)  -> flattened to (N,P)
- x:     (N,G) or (...,G)  -> flattened to (N,G)

Outputs saved in --run_dir:
- posterior.pt
- density_estimator.pt
- train_manifest.txt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch
from sbi.inference import SNPE_C
from sbi.utils.torchutils import BoxUniform
from sbi.neural_nets import posterior_nn


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _flatten_lastdim(a: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,D) or (...,D) into (N,D) by flattening leading dims.
    """
    if a.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims (...,D). Got {tuple(a.shape)}")
    D = a.shape[-1]
    return a.reshape(-1, D)


def _load_tensor(path: str, device: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.to(device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Output folder for this training run.")
    ap.add_argument("--theta_path", type=str, required=True, help="Path to theta.pt (or .pth).")
    ap.add_argument("--x_path", type=str, required=True, help="Path to x.pt (or .pth).")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Estimator config
    ap.add_argument("--estimator", type=str, default="nsf", choices=["nsf", "mdn"])
    ap.add_argument("--num_transforms", type=int, default=8, help="NSF: num_transforms, MDN: num_components")
    ap.add_argument("--training_batch_size", type=int, default=4096)
    ap.add_argument("--max_num_epochs", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=5e-4)

    # Misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_threads", type=int, default=4)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    run_dir = Path(args.run_dir)
    _ensure_dir(run_dir)

    device = args.device

    # -------------------------
    # Load + flatten
    # -------------------------
    theta = _load_tensor(args.theta_path, device=device)
    x = _load_tensor(args.x_path, device=device)

    theta2d = _flatten_lastdim(theta)
    x2d = _flatten_lastdim(x)

    if theta2d.shape[0] != x2d.shape[0]:
        raise ValueError(f"Sample mismatch: theta N={theta2d.shape[0]} vs x N={x2d.shape[0]}")

    # Train in float32 (even if stored float16)
    theta2d = theta2d.float()
    x2d = x2d.float()

    # -------------------------
    # Prior placeholder (dimension match)
    # -------------------------
    theta_min = theta2d.min(dim=0).values
    theta_max = theta2d.max(dim=0).values
    prior = BoxUniform(low=theta_min, high=theta_max)

    # -------------------------
    # Build density estimator
    # -------------------------
    if args.estimator == "nsf":
        net = posterior_nn(model="nsf", num_transforms=args.num_transforms)
    else:
        net = posterior_nn(model="mdn", num_components=args.num_transforms)

    # -------------------------
    # Train SNPE-C
    # -------------------------
    trainer = SNPE_C(
        prior=prior,
        density_estimator=net,
        device=device,
        show_progress_bars=True,
        logging_level="INFO",
    )

    density_estimator = trainer.append_simulations(theta2d, x2d).train(
        training_batch_size=args.training_batch_size,
        max_num_epochs=args.max_num_epochs,
        learning_rate=args.learning_rate,
        show_train_summary=True,
        use_combined_loss=True,
    )

    posterior = trainer.build_posterior(density_estimator)

    # -------------------------
    # Save artifacts
    # -------------------------
    torch.save(posterior, run_dir / "posterior.pt")
    torch.save(density_estimator, run_dir / "density_estimator.pt")

    manifest = {
        "theta_path": args.theta_path,
        "x_path": args.x_path,
        "theta_shape_in": tuple(theta.shape),
        "x_shape_in": tuple(x.shape),
        "theta_shape_2d": tuple(theta2d.shape),
        "x_shape_2d": tuple(x2d.shape),
        "device": device,
        "estimator": args.estimator,
        "num_transforms_or_components": args.num_transforms,
        "training_batch_size": args.training_batch_size,
        "max_num_epochs": args.max_num_epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }
    (run_dir / "train_manifest.txt").write_text("\n".join([f"{k}: {v}" for k, v in manifest.items()]) + "\n")

    print(f"[OK] Saved posterior to: {run_dir / 'posterior.pt'}")
    print(f"[OK] Saved density_estimator to: {run_dir / 'density_estimator.pt'}")


if __name__ == "__main__":
    main()