#!/usr/bin/env python3
"""
simulate_data.py

Standalone simulator runner for Ball-and-Sticks attenuation model.

Why this script exists
----------------------
You want three workflows that can be called independently from terminal:
  (A) simulate data
  (B) train
  (C) inference

This script is (A). It focuses ONLY on generating theta and x (noisefree/noisy),
with options that help "paper-style sanity checks" where theta is a *volume*.

Key capabilities
----------------
1) Theta can be provided as:
     - (N, P)           typical SBI training samples
     - (X, Y, Z, P)     handcrafted parameter grids / test volumes
     - (..., P)         any leading dims are supported
   Internally we flatten to (N, P), simulate, then reshape back to (..., G).

2) Noise is applied OUTSIDE the simulator:
     - simulator(theta_omega) -> x_noisefree
     - apply_noise_policy(x_noisefree, NoiseConfig) -> x_noisy (+ snr_used)
   This matches your design choice: noise is modular and reusable.

3) If theta includes SNR as last column (paper-style joint theta=[Omega,SNR]):
     - set --include_snr_in_theta
     - the simulator will be fed theta[:, :-1]
   (Noise generation can still be controlled independently via CLI.)

4) Saving:
     - theta.pt, x_noisefree.pt, x_noisy.pt, snr_used.pt (torch tensors)
     - optional NIfTI export if you pass --export_nifti and --ref_nifti:
         * theta.nii.gz (multi-volume)
         * theta_params/*.nii.gz (each parameter separately)
         * x_noisefree.nii.gz, x_noisy.nii.gz

Important constraint re multilevel noise
----------------------------------------
Noise strategy="multilevel" repeats the same noisefree sample across multiple SNR bins.
That changes the number of samples. Therefore it cannot preserve a fixed 3D theta grid
shape (X,Y,Z,...) without ambiguity. So we block multilevel if theta is volumetric.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from tools.io_nifti import load_any_array, export_any_to_nifti
from tools.noise import NoiseConfig, apply_noise_policy

from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors
from models.ball_and_sticks.simulator import GradientTable, BallAndSticksAttenuation

# NEW: export each parameter as its own NIfTI
from models.ball_and_sticks.export_params import export_theta_params_to_nifti


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _flatten_theta(theta: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Accept theta as (..., P). Flatten to (N, P) for simulation.

    Returns
    -------
    theta2d : (N, P)
    lead_shape : (...) original leading shape (excluding P)
    """
    if theta.ndim < 2:
        raise ValueError(f"theta must have at least 2 dims (...,P). Got {tuple(theta.shape)}")
    P = theta.shape[-1]
    lead_shape = tuple(theta.shape[:-1])
    theta2d = theta.reshape(-1, P)
    return theta2d, lead_shape


def _reshape_x(x2d: torch.Tensor, lead_shape: Tuple[int, ...], G: int) -> torch.Tensor:
    """
    Reshape simulated signals back to match theta's leading dims.

    If theta was (X,Y,Z,P), x becomes (X,Y,Z,G).
    """
    return x2d.reshape(*lead_shape, G)


def build_ball_and_sticks_simulator(
    bvals_path: str,
    bvecs_path: str,
    device: str = "cpu",
) -> BallAndSticksAttenuation:
    """
    Build the attenuation simulator from (bvals,bvecs) with b0 removed.

    This is intentionally simple; all model complexity is in BallAndSticksAttenuation.
    """
    gtab = GradientTable.read_bvals_bvecs(bvals_path, bvecs_path, device=device)
    return BallAndSticksAttenuation(gtab=gtab, device=device)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # Output
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for this simulation run.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)

    # Acquisition / model
    ap.add_argument("--bvals", type=str, required=True, help="bvals (NO b0s) file.")
    ap.add_argument("--bvecs", type=str, required=True, help="bvecs (NO b0s) file.")
    ap.add_argument("--nfib", type=int, default=3)
    ap.add_argument("--modelnum", type=int, default=2)
    ap.add_argument(
        "--hemisphere",
        action="store_true",
        default=True,
        help="If set, phi is restricted to hemisphere (legacy convention).",
    )

    # Theta input
    ap.add_argument(
        "--theta_path",
        type=str,
        default=None,
        help="Optional: path to theta samples (.pt/.npy/.nii.gz). Shape (...,P).",
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="If no theta_path: number of samples to draw from the prior.",
    )

    # Theta includes SNR?
    ap.add_argument(
        "--include_snr_in_theta",
        action="store_true",
        default=False,
        help=(
            "If True, provided theta is assumed to be [Omega, SNR] with SNR last. "
            "Simulator will ignore SNR by stripping last column before simulation. "
            "This is useful if you reuse joint-theta tensors from training/inference."
        ),
    )

    # Noise controls (applied after noisefree simulation)
    ap.add_argument("--noise_enabled", action="store_true", default=False)
    ap.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "rician"])
    ap.add_argument("--noise_strategy", type=str, default="random", choices=["random", "fixed", "multilevel"])
    ap.add_argument("--snr_min", type=float, default=3.0)
    ap.add_argument("--snr_max", type=float, default=80.0)
    ap.add_argument("--n_levels", type=int, default=8)
    ap.add_argument("--snr_fixed", type=float, default=30.0)
    ap.add_argument("--snr_fixed_jitter", type=float, default=0.0)

    # Saving / exports
    ap.add_argument(
        "--float16",
        action="store_true",
        default=False,
        help="Store tensors as float16 to save disk. (Note: noise + exp may prefer float32 for numerics.)",
    )
    ap.add_argument(
        "--save_noisefree",
        action="store_true",
        default=True,
        help="Save x_noisefree (True by default; very useful for sanity checks).",
    )
    ap.add_argument("--export_nifti", action="store_true", default=False)
    ap.add_argument("--ref_nifti", type=str, default=None, help="Reference NIfTI for affine/header when exporting.")

    args = ap.parse_args()

    # Determinism (best-effort; CUDA will still be non-deterministic in some ops)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    device = args.device

    # ---------------------------------------------------------------------
    # Build simulator
    # ---------------------------------------------------------------------
    simulator = build_ball_and_sticks_simulator(args.bvals, args.bvecs, device=device)
    G = simulator.gtab.bvals.numel()  # number of gradients (b0 already removed)

    # ---------------------------------------------------------------------
    # Load or sample theta
    # ---------------------------------------------------------------------
    if args.theta_path is not None:
        # Supports .pt/.npy/.nii.gz via your tools.io_nifti.load_any_array
        arr = load_any_array(args.theta_path)
        theta = torch.as_tensor(arr, dtype=torch.float32, device=device)
    else:
        # Sample from prior
        # Note: include_snr_in_theta affects ONLY the prior sampling here.
        # If you want Omega-only priors but still noise later, keep include_snr_in_theta False.
        prior_cfg = BallAndSticksPriorConfig(
            nfib=args.nfib,
            modelnum=args.modelnum,
            hemisphere=args.hemisphere,
            device=device,
            include_snr=args.include_snr_in_theta,
        )
        prior = build_ball_and_sticks_priors(prior_cfg)
        theta = prior.sample((args.n_samples,))  # (N,P) or (N,P+1) if include_snr_in_theta

    theta2d, lead_shape = _flatten_theta(theta)

    # ---------------------------------------------------------------------
    # If theta contains SNR, strip it before calling the simulator
    # ---------------------------------------------------------------------
    if args.include_snr_in_theta:
        if theta2d.shape[1] < 2:
            raise ValueError("theta has too few params to include SNR.")
        theta_omega = theta2d[:, :-1]
    else:
        theta_omega = theta2d

    # ---------------------------------------------------------------------
    # Simulate noisefree signals: (N,G) then reshape to (...,G)
    # ---------------------------------------------------------------------
    x0_2d = simulator(theta_omega, nfib=args.nfib, modelnum=args.modelnum)
    x0 = _reshape_x(x0_2d, lead_shape, G)

    # ---------------------------------------------------------------------
    # Apply noise policy (optional)
    # ---------------------------------------------------------------------
    x_noisy = None
    snr_used = None
    cfg = None  # for manifest writing

    if args.noise_enabled:
        cfg = NoiseConfig(
            noise_type=args.noise_type,
            strategy=args.noise_strategy,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            n_levels=args.n_levels,
            snr_fixed=args.snr_fixed,
            snr_fixed_jitter=args.snr_fixed_jitter,
            device=device,
        )

        # Multilevel repeats samples => cannot preserve a 3D theta grid
        # lead_shape is theta.shape[:-1]; if it's volumetric (e.g. X,Y,Z), we block it.
        if cfg.strategy == "multilevel" and len(lead_shape) > 1:
            raise ValueError(
                "noise_strategy='multilevel' repeats samples and breaks volume-shaped theta.\n"
                "Use 'random' or 'fixed' for (X,Y,Z,P) sanity-check volumes."
            )

        # apply_noise_policy expects (N,G)
        x0_flat = x0.reshape(-1, G)
        x_noisy_flat, snr_used = apply_noise_policy(x0_flat, cfg, base_repeat=True)
        x_noisy = x_noisy_flat.reshape(*lead_shape, G)

    # ---------------------------------------------------------------------
    # Save tensors (torch) in a single chunk (your requested “old style”)
    # ---------------------------------------------------------------------
    save_dtype = torch.float16 if args.float16 else torch.float32

    torch.save(theta.detach().to("cpu", dtype=save_dtype), out_dir / "theta.pt")

    if args.save_noisefree:
        torch.save(x0.detach().to("cpu", dtype=save_dtype), out_dir / "x_noisefree.pt")

    if x_noisy is not None:
        torch.save(x_noisy.detach().to("cpu", dtype=save_dtype), out_dir / "x_noisy.pt")

    if snr_used is not None:
        # SNR is cheap; keep float32 for clarity even if signals are float16
        torch.save(snr_used.detach().to("cpu", dtype=torch.float32), out_dir / "snr_used.pt")

    # ---------------------------------------------------------------------
    # Optional NIfTI export (geometry preserved via ref_nifti)
    # ---------------------------------------------------------------------
    if args.export_nifti:
        if args.ref_nifti is None:
            raise ValueError("--export_nifti requires --ref_nifti to keep geometry consistent.")

        # Full theta as 4D/5D NIfTI (depending on your input dims)
        export_any_to_nifti(theta.detach().to("cpu"), out_dir / "theta.nii.gz", ref_nifti=args.ref_nifti, dtype="float32")

        # NEW: export each parameter as its own NIfTI in theta_params/
        # This makes it easy to visually inspect parameter grids or ground-truth maps.
        export_theta_params_to_nifti(
            theta.detach().to("cpu"),
            out_dir / "theta_params",
            nfib=args.nfib,
            modelnum=args.modelnum,
            include_snr=args.include_snr_in_theta,
            ref_nifti=args.ref_nifti,
            dtype=(np.float16 if args.float16 else np.float32),
            prefix="",
            overwrite=True,
        )

        # Signals
        if args.save_noisefree:
            export_any_to_nifti(x0.detach().to("cpu"), out_dir / "x_noisefree.nii.gz", ref_nifti=args.ref_nifti, dtype="float32")
        if x_noisy is not None:
            export_any_to_nifti(x_noisy.detach().to("cpu"), out_dir / "x_noisy.nii.gz", ref_nifti=args.ref_nifti, dtype="float32")

    # ---------------------------------------------------------------------
    # Write a run manifest (plain text, greppable)
    # ---------------------------------------------------------------------
    manifest = {
        "seed": args.seed,
        "device": device,
        "nfib": args.nfib,
        "modelnum": args.modelnum,
        "G": int(G),
        "theta_path": args.theta_path,
        "theta_shape": tuple(theta.shape),
        "x0_shape": tuple(x0.shape),
        "noise_enabled": args.noise_enabled,
        "noise_cfg": asdict(cfg) if cfg is not None else None,
        "float16": args.float16,
        "saved_noisefree": args.save_noisefree,
        "export_nifti": args.export_nifti,
        "ref_nifti": args.ref_nifti,
        "include_snr_in_theta": args.include_snr_in_theta,
    }
    (out_dir / "manifest.txt").write_text("\n".join([f"{k}: {v}" for k, v in manifest.items()]) + "\n")


if __name__ == "__main__":
    main()