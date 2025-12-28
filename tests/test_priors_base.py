"""
tests/test_priors_base.py

Tests for low-level prior distributions in priors/base.py.

These tests are BOTH:
- numerical sanity checks (assertions)
- visual sanity checks (histograms)

Default behaviour:
- Save plots to ./test_plots/ (works on headless/SSH/HPC)
Optional:
- To also show plots interactively, run with:
    SHOW_PLOTS=1 pytest -s tests/test_priors_base.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from priors.base import (
    CorrectedThetaUniform,
    PhiUniformHemisphere,
    PhiUniformSphere,
    ARDPrior,
)

# Keep this large for smooth histograms, but you can override:
#   N_SAMPLES=50000 pytest -s tests/test_priors_base.py
N_SAMPLES = int(os.environ.get("N_SAMPLES", "200000"))

# Where to save plots (override with TEST_PLOTS_DIR=/path)
PLOTS_DIR = Path(os.environ.get("TEST_PLOTS_DIR", "test_plots"))

# If SHOW_PLOTS=1, also pop up figures (only works with GUI backend)
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"


def _finish_fig(fig: plt.Figure, name: str) -> None:
    """Save figure and optionally display it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PLOTS_DIR / f"{name}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _hist(samples: np.ndarray, title: str, xlabel: str, name: str, bins: int = 100) -> None:
    """Convenience histogram plot."""
    fig = plt.figure()
    plt.hist(samples, bins=bins, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    _finish_fig(fig, name=name)


def test_corrected_theta_uniform():
    """
    Theta prior should sample theta ∈ [0, pi] such that cos(theta) ~ Uniform[-1, 1].
    """
    prior = CorrectedThetaUniform(device="cpu")
    theta = prior.sample((N_SAMPLES,)).cpu().numpy().squeeze()

    # Shape + range checks
    assert theta.ndim == 1
    assert np.all(theta >= 0.0)
    assert np.all(theta <= np.pi)

    cos_theta = np.cos(theta)

    # Distribution sanity checks
    # For cos(theta) uniform on [-1,1], mean should be ~0.
    assert abs(np.mean(cos_theta)) < 0.01

    # Also check roughly uniform variance: Var(U[-1,1]) = 1/3 ≈ 0.333...
    assert abs(np.var(cos_theta) - (1.0 / 3.0)) < 0.02

    # Plots
    _hist(
        theta,
        title="CorrectedThetaUniform: theta ∈ [0, π] (sphere-correct)",
        xlabel="theta (rad)",
        name="theta_corrected",
    )
    _hist(
        cos_theta,
        title="CorrectedThetaUniform: cos(theta) ∈ [-1, 1] (should be uniform)",
        xlabel="cos(theta)",
        name="cos_theta_uniform",
    )


def test_phi_uniform_hemisphere():
    """
    PhiUniformHemisphere should sample phi ∈ [0, pi] uniformly.
    """
    prior = PhiUniformHemisphere(device="cpu")
    phi = prior.sample((N_SAMPLES,)).cpu().numpy().squeeze()

    assert phi.ndim == 1
    assert np.all(phi >= 0.0)
    assert np.all(phi <= np.pi)

    # Uniform(0, π) mean is π/2
    assert abs(np.mean(phi) - np.pi / 2) < 0.02

    # Uniform(0, π) variance is π^2 / 12
    assert abs(np.var(phi) - (np.pi**2) / 12.0) < 0.05

    _hist(
        phi,
        title="PhiUniformHemisphere: phi ∈ [0, π] (uniform)",
        xlabel="phi (rad)",
        name="phi_hemisphere",
    )


def test_phi_uniform_sphere():
    """
    PhiUniformSphere should sample phi ∈ [0, 2pi] uniformly.
    """
    prior = PhiUniformSphere(device="cpu")
    phi = prior.sample((N_SAMPLES,)).cpu().numpy().squeeze()

    assert phi.ndim == 1
    assert np.all(phi >= 0.0)
    assert np.all(phi <= 2 * np.pi)

    # Uniform(0, 2π) mean is π
    assert abs(np.mean(phi) - np.pi) < 0.02

    # Uniform(0, 2π) variance is (2π)^2 / 12 = π^2/3
    assert abs(np.var(phi) - (np.pi**2) / 3.0) < 0.08

    _hist(
        phi,
        title="PhiUniformSphere: phi ∈ [0, 2π] (uniform)",
        xlabel="phi (rad)",
        name="phi_sphere",
    )


def test_ard_prior():
    """
    ARDPrior should behave like a bounded log-uniform distribution,
    which corresponds to p(x) ∝ 1/x on [min_val, max_val].
    """
    min_val, max_val = 1e-3, 1.0
    prior = ARDPrior(min_val=min_val, max_val=max_val, device="cpu")
    x = prior.sample((N_SAMPLES,)).cpu().numpy().squeeze()

    assert x.ndim == 1
    assert np.all(x >= min_val)
    assert np.all(x <= max_val)

    # If x is log-uniform, then log(x) should be uniform.
    logx = np.log(x)
    a, b = np.log(min_val), np.log(max_val)

    # Uniform(a,b) mean is (a+b)/2
    assert abs(np.mean(logx) - (a + b) / 2.0) < 0.02

    # Uniform(a,b) variance is (b-a)^2/12
    assert abs(np.var(logx) - ((b - a) ** 2) / 12.0) < 0.08

    _hist(
        x,
        title="ARDPrior samples (bounded): p(x) ∝ 1/x",
        xlabel="x",
        name="ard_x",
    )
    _hist(
        logx,
        title="ARDPrior: log(x) should be uniform",
        xlabel="log(x)",
        name="ard_logx",
    )