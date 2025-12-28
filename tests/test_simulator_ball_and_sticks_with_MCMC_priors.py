# pytest -q -s tests/test_simulator_ball_and_sticks_with_MCMC_priors.py

"""
tests/test_simulator_ball_and_sticks_with_priors.py

Purpose
-------
Integration + sanity tests connecting:
    Ball&Sticks prior -> Ball&Sticks simulator

Key point
---------
The current "simple MCMC-style" priors sample f1,f2,f3 independently.
That means sum(f) can exceed 1 (invalid mixture weights).
When sum(f) > 1, the isotropic weight (1 - sum(f)) becomes negative and
the simulated attenuation can exceed 1.

This file makes that explicit by:
1) Printing how often sum(f) > 1 happens under the current priors.
2) Testing strict physical bounds only on the subset with sum(f) <= 1.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from utils_plotting import _save_example_signals
from models.ball_and_sticks.simulator import GradientTable, BallAndSticksAttenuation
from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors


def _random_unit_bvecs(G: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, G)).astype(np.float32)
    v /= np.linalg.norm(v, axis=0, keepdims=True) + 1e-12
    return v


def _make_gtab(G: int = 60, bval: float = 1000.0) -> GradientTable:
    bvals = np.full((G,), bval, dtype=np.float32)
    bvecs = _random_unit_bvecs(G)
    return GradientTable(
        bvals=torch.tensor(bvals, dtype=torch.float32),
        bvecs=torch.tensor(bvecs, dtype=torch.float32),
    )


def _fraction_indices_for_nfib(nfib: int) -> list[int]:
    """
    Ω layout is:
      [ d,
        f1, th1, ph1,
        f2, th2, ph2,
        f3, th3, ph3,
        (d_std if modelnum=2)
      ]
    So the f_i indices are: 1, 4, 7 for nfib=3, truncated for nfib<3.
    """
    return [1 + 3 * i for i in range(nfib)]


def test_prior_to_simulator_runs_end_to_end_model2():
    """
    End-to-end integration test:
        Ω ~ prior  ->  simulator(Ω)

    We do NOT require y <= 1 here because (with independent f_i priors)
    sum(f) can exceed 1, and that can produce y > 1.

    Instead, we:
      - check shape
      - check finiteness
      - print a warning if sum(f) > 1 occurs (expected with current priors)
    """
    gtab = _make_gtab(G=48, bval=1200.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    cfg = BallAndSticksPriorConfig(nfib=3, modelnum=2, device="cpu", hemisphere=True)
    prior = build_ball_and_sticks_priors(cfg)

    N = 512
    theta = prior.sample((N,)).float()

    # ---- show how often the "traditional" priors violate mixture constraints ----
    f_idx = _fraction_indices_for_nfib(cfg.nfib)
    fsum = torch.zeros((N,), dtype=torch.float32)
    for j in f_idx:
        fsum += theta[:, j]

    invalid = fsum > 1.0
    n_invalid = int(torch.sum(invalid).item())
    frac_invalid = n_invalid / N

    if n_invalid > 0:
        # This is the educational / reproducibility message you asked for.
        max_fsum = float(torch.max(fsum).item())
        print(
            "\n[WARNING] Unconstrained fraction priors produced invalid mixtures: "
            f"sum(f)>1 for {n_invalid}/{N} samples ({100*frac_invalid:.1f}%). "
            f"Max sum(f) observed: {max_fsum:.3f}. "
            "This can lead to attenuation > 1 because (1-sum(f)) becomes negative.\n"
        )
    else:
        print("\n[INFO] No invalid mixtures found in this draw (sum(f)>1 = 0). This is rare but possible.\n")

    # ---- simulate ----
    y = sim(theta, nfib=cfg.nfib, modelnum=cfg.modelnum)

    assert y.shape == (N, gtab.bvals.numel())
    assert torch.isfinite(y).all()

    # With invalid mixtures, y can exceed 1. But it should not explode numerically.
    assert torch.max(y) <= 2.0
    assert torch.min(y) >= -1e-6


def test_if_fsum_le_1_then_attenuation_is_bounded_by_one():
    """
    Strict physical sanity check:
    If sum(f) <= 1, then attenuation should be in [0, 1] (up to tiny tolerance).
    """
    gtab = _make_gtab(G=48, bval=1200.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    cfg = BallAndSticksPriorConfig(nfib=3, modelnum=2, device="cpu", hemisphere=True)
    prior = build_ball_and_sticks_priors(cfg)

    N = 5000
    theta = prior.sample((N,)).float()

    f_idx = _fraction_indices_for_nfib(cfg.nfib)
    fsum = torch.zeros((N,), dtype=torch.float32)
    for j in f_idx:
        fsum += theta[:, j]

    valid = fsum <= 1.0
    n_valid = int(torch.sum(valid).item())

    if n_valid == 0:
        # If this ever happens, the priors are very broken / too wide.
        raise AssertionError("No samples with sum(f) <= 1 found. Priors likely incorrect.")

    # Small message so the user understands what's being tested
    print(f"\n[INFO] Testing physical bounds only on valid mixtures: {n_valid}/{N} samples have sum(f)<=1.\n")

    y = sim(theta[valid], nfib=cfg.nfib, modelnum=cfg.modelnum)

    assert torch.isfinite(y).all()
    assert torch.max(y) <= 1.0 + 1e-6
    assert torch.min(y) >= 0.0 - 1e-8


def test_prior_to_simulator_runs_end_to_end_model2():
    gtab = _make_gtab(G=48, bval=1200.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    cfg = BallAndSticksPriorConfig(nfib=3, modelnum=2, device="cpu")
    prior = build_ball_and_sticks_priors(cfg)

    N = 512
    theta = prior.sample((N,)).float()

    # --- compute sum(f) ---
    f_idx = _fraction_indices_for_nfib(cfg.nfib)
    fsum = torch.zeros((N,), dtype=torch.float32)
    for j in f_idx:
        fsum += theta[:, j]

    invalid = fsum > 1.0
    n_invalid = int(invalid.sum())

    if n_invalid > 0:
        print(
            f"[WARNING] sum(f) > 1 for {n_invalid}/{N} samples "
            f"({100*n_invalid/N:.1f}%), "
            f"max = {fsum.max():.3f}"
        )

    y = sim(theta, nfib=cfg.nfib, modelnum=cfg.modelnum)

    assert y.shape == (N, gtab.bvals.numel())
    assert torch.isfinite(y).all()
    assert torch.max(y) <= 2.0  # loose, intentional

    # --- save plots ---
    if n_invalid > 0:
        _save_example_signals(
            y[invalid][:10],
            title="Ball-and-Sticks (INVALID mixtures: sum(f) > 1)",
            filename="simulator_prior_invalid.png",
        )

    _save_example_signals(
        y[~invalid][:10],
        title="Ball-and-Sticks (valid mixtures: sum(f) ≤ 1)",
        filename="simulator_prior_valid.png",
    )