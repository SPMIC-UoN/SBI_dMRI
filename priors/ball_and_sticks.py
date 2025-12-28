"""
priors/ball_and_sticks.py

Model-specific prior assembly for the Ball-and-Sticks attenuation model.

This module defines the Ω parameterization and builds a prior distribution
object compatible with sbi.

Ω layout (non-restricted, simple priors)
----------------------------------------
For nfib fibres:

modelnum = 1 (single-shell):
    Ω = [ d,
          f1, th1, phi1,
          f2, th2, phi2,
          ...
        ]
    dim = 1 + 3*nfib

modelnum = 2 (multi-shell; includes dispersion parameter d_std):
    Ω = [ d,
          f1, th1, phi1,
          ...
          d_std
        ]
    dim = 2 + 3*nfib

Optional SNR augmentation (noise-aware amortisation)
---------------------------------------------------
If include_snr=True, we append SNR as the LAST parameter:

    θ = [ Ω, SNR ]
    dim = omega_dim(...) + 1

This allows the NPE (SNPE-C here) to amortise across noise levels and also
estimate SNR during inference. This matches the paper setup where we
explicitly concatenate SNR into θ.

Important note about priors vs training
---------------------------------------
In single-round SNPE(-C), the prior is mainly used to generate training θ.
Once you have (θ, x), training just learns the mapping θ -> x (via NPE loss).
However, sbi still *stores the prior* in the posterior object, and it will
validate θ dimensionality / support. Therefore, if θ includes SNR, the prior
MUST include SNR too (otherwise shape mismatch errors can occur, often at
posterior building or sampling time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from torch.distributions import Distribution
from sbi.utils.user_input_checks_utils import MultipleIndependent

from priors.base import (
    box_uniform_1d,
    gamma_1d,
    CorrectedThetaUniform,
    PhiUniformHemisphere,
    PhiUniformSphere,
)


@dataclass(frozen=True)
class BallAndSticksPriorConfig:
    # Core
    nfib: int = 3
    modelnum: int = 2
    device: str = "cpu"

    # Orientation domain
    hemisphere: bool = True  # True: phi in [0,pi]; False: phi in [0,2pi]

    # Diffusivity priors
    use_gamma_d: bool = False
    d_uniform: Tuple[float, float] = (1e-5, 5e-3)
    d_gamma: Tuple[float, float] = (2.52, 1500.0)  # (shape, rate)

    # d_std priors (only modelnum=2)
    use_gamma_d_std: bool = False
    d_std_uniform: Tuple[float, float] = (0.0, 5e-3)
    d_std_gamma: Tuple[float, float] = (0.4, 600.0)

    # Volume fraction bounds (simple MCMC-style; not restricted)
    f1: Tuple[float, float] = (0.0, 0.9)
    f2: Tuple[float, float] = (0.05, 0.5)
    f3: Tuple[float, float] = (0.05, 0.33)

    # ---- Noise / SNR augmentation (paper-style) ----
    include_snr: bool = False
    snr_uniform: Tuple[float, float] = (2.0, 80.0)  # you can set (0,80) if you really want


def _validate_cfg(cfg: BallAndSticksPriorConfig) -> None:
    if cfg.nfib not in (1, 2, 3):
        raise ValueError("nfib must be 1, 2, or 3.")
    if cfg.modelnum not in (1, 2):
        raise ValueError("modelnum must be 1 or 2.")
    if cfg.modelnum == 1 and cfg.use_gamma_d_std:
        raise ValueError("d_std is not used for modelnum=1 (single-shell).")
    if cfg.include_snr:
        lo, hi = cfg.snr_uniform
        if lo < 0 or hi <= lo:
            raise ValueError("snr_uniform must satisfy 0 <= lo < hi.")


def _fraction_priors(cfg: BallAndSticksPriorConfig) -> List[Distribution]:
    # f2/f3 bounds are irrelevant if nfib < 2/3, but harmless to keep.
    return [
        box_uniform_1d(cfg.f1[0], cfg.f1[1], device=cfg.device),
        box_uniform_1d(cfg.f2[0], cfg.f2[1], device=cfg.device),
        box_uniform_1d(cfg.f3[0], cfg.f3[1], device=cfg.device),
    ]


def build_ball_and_sticks_priors(cfg: BallAndSticksPriorConfig) -> Distribution:
    """
    Build a simple (non-restricted) prior for Ball-and-Sticks θ.

    Returns
    -------
    torch.distributions.Distribution
        A MultipleIndependent distribution that samples θ vectors.
    """
    _validate_cfg(cfg)

    # d prior
    prior_d = (
        gamma_1d(cfg.d_gamma[0], cfg.d_gamma[1], device=cfg.device)
        if cfg.use_gamma_d
        else box_uniform_1d(cfg.d_uniform[0], cfg.d_uniform[1], device=cfg.device)
    )

    # orientation priors
    prior_th = CorrectedThetaUniform(device=cfg.device)
    prior_phi = PhiUniformHemisphere(device=cfg.device) if cfg.hemisphere else PhiUniformSphere(device=cfg.device)

    # fraction priors
    f_priors = _fraction_priors(cfg)

    priors_list: List[Distribution] = [prior_d]

    # Per-fibre: [fi, thi, phii]
    for i in range(cfg.nfib):
        priors_list.extend([f_priors[i], prior_th, prior_phi])

    # modelnum=2 adds d_std
    if cfg.modelnum == 2:
        prior_d_std = (
            gamma_1d(cfg.d_std_gamma[0], cfg.d_std_gamma[1], device=cfg.device)
            if cfg.use_gamma_d_std
            else box_uniform_1d(cfg.d_std_uniform[0], cfg.d_std_uniform[1], device=cfg.device)
        )
        priors_list.append(prior_d_std)

    # Optional: append SNR at the end to match paper setup theta=[Ω, SNR]
    if cfg.include_snr:
        priors_list.append(box_uniform_1d(cfg.snr_uniform[0], cfg.snr_uniform[1], device=cfg.device))

    return MultipleIndependent(priors_list)


def omega_dim(nfib: int, modelnum: int) -> int:
    """Return expected dimensionality of Ω (no SNR)."""
    if modelnum == 1:
        return 1 + 3 * nfib
    if modelnum == 2:
        return 2 + 3 * nfib
    raise ValueError("modelnum must be 1 or 2.")


def theta_dim(nfib: int, modelnum: int, include_snr: bool) -> int:
    """Return expected dimensionality of θ = Ω (+ optional SNR)."""
    return omega_dim(nfib, modelnum) + (1 if include_snr else 0)