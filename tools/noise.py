"""
tools/noise.py

Noise utilities for diffusion MRI (or any) simulated signals.

We assume signals are *attenuation* S/S0 so that:
- noisefree signal values are typically in [0, 1]
- we define SNR as: sigma = 1 / SNR

Noise is applied OUTSIDE the forward model so the same noisefree sample can be
reused with multiple noise levels (useful for training), e.g. the "multilevel"
strategy from your paper.

Supported noise types
---------------------
- "gaussian": additive Gaussian noise
- "rician": magnitude (Rician) noise by adding noise in quadrature

Supported SNR strategies
------------------------
- "random":     one random SNR per noisefree sample
- "multilevel": replicate noisefree samples across multiple SNR intervals
- "fixed":      same SNR for all samples (optional jitter)

Notes
-----
- All functions are pure torch (no numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch


NoiseType = Literal["gaussian", "rician"]
NoiseStrategy = Literal["random", "multilevel", "fixed"]


# -----------------------------------------------------------------------------
# Core noise
# -----------------------------------------------------------------------------
def _as_1d_signal(x: torch.Tensor) -> torch.Tensor:
    """Flatten a signal into 1D for noise application, preserving total length."""
    if x.ndim == 1:
        return x
    return x.reshape(-1)


def _as_scalar_tensor(x: torch.Tensor | float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert float/tensor to a scalar tensor on the right device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(x), device=device, dtype=dtype).reshape(())


def add_noise(
    x: torch.Tensor,
    snr: torch.Tensor | float,
    noise_type: NoiseType = "gaussian",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Add noise to a single signal.

    Parameters
    ----------
    x
        Signal tensor. Shape (G,) or any shape; will be flattened internally
        then reshaped back to original shape.
    snr
        Signal-to-noise ratio value. We use sigma = 1 / (snr + eps).
        Can be float or tensor (kept tensor-safe for GPU later).
    noise_type
        "gaussian" or "rician".
    eps
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Noisy signal with the same shape as x.
    """
    x_in = x
    x_flat = _as_1d_signal(x).to(dtype=torch.float32)

    snr_t = _as_scalar_tensor(snr, device=x_flat.device, dtype=x_flat.dtype)
    if torch.any(snr_t <= 0):
        raise ValueError(f"SNR must be > 0. Got {snr_t.item()}.")

    sigma = 1.0 / (snr_t + eps)

    if noise_type == "gaussian":
        noise = torch.empty_like(x_flat).normal_(mean=0.0, std=float(sigma.item()))
        y = x_flat + noise

    elif noise_type == "rician":
        n1 = torch.empty_like(x_flat).normal_(mean=0.0, std=float(sigma.item()))
        n2 = torch.empty_like(x_flat).normal_(mean=0.0, std=float(sigma.item()))
        y = torch.sqrt((x_flat + n1).pow(2) + n2.pow(2))

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return y.reshape(x_in.shape)


def add_noise_batch(
    x: torch.Tensor,
    snr_values: torch.Tensor,
    noise_type: NoiseType = "gaussian",
) -> torch.Tensor:
    """
    Add noise to a batch of signals.

    Parameters
    ----------
    x
        Tensor of noisefree signals. Shape (N, G). If shape (G,), treated as N=1.
    snr_values
        SNR values. Shape (N,) or (N,1).
    noise_type
        "gaussian" or "rician".

    Returns
    -------
    torch.Tensor
        Noisy signals of shape (N, G).
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    snr_values = snr_values.reshape(-1).to(device=x.device, dtype=torch.float32)
    if x.shape[0] != snr_values.shape[0]:
        raise ValueError(f"Batch mismatch: x has N={x.shape[0]}, snr_values has N={snr_values.shape[0]}.")

    ys = []
    for i in range(x.shape[0]):
        ys.append(add_noise(x[i], snr_values[i], noise_type=noise_type))
    return torch.stack(ys, dim=0)


# -----------------------------------------------------------------------------
# SNR sampling strategies
# -----------------------------------------------------------------------------
def sample_snr_random(
    n: int,
    snr_min: float = 2.0,
    snr_max: float = 80.0,
    device: str = "cpu",
) -> torch.Tensor:
    """One random SNR per sample."""
    if snr_min <= 0 or snr_max <= 0 or snr_max <= snr_min:
        raise ValueError("Require 0 < snr_min < snr_max.")
    return torch.empty((n,), device=device).uniform_(snr_min, snr_max)


def sample_snr_multilevel(
    n_base: int,
    snr_min: float = 2.0,
    snr_max: float = 80.0,
    n_levels: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Multilevel SNR sampling: create one SNR per (base-sample, level) by sampling
    within each interval of [snr_min, snr_max].

    Returns SNR of length n_base * n_levels, ordered in blocks by level:
      [level0 for all base] [level1 for all base] ...
    (shuffle later if desired)
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    if snr_min <= 0 or snr_max <= 0 or snr_max <= snr_min:
        raise ValueError("Require 0 < snr_min < snr_max.")

    edges = torch.linspace(snr_min, snr_max, n_levels + 1, device=device)
    snrs = []
    for i in range(n_levels):
        lo = float(edges[i].item())
        hi = float(edges[i + 1].item())
        snrs.append(torch.empty((n_base,), device=device).uniform_(lo, hi))
    return torch.cat(snrs, dim=0)


def sample_snr_fixed(
    n: int,
    snr_value: float = 30.0,
    jitter: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Fixed SNR for all samples, optionally with small uniform jitter."""
    if snr_value <= 0:
        raise ValueError("snr_value must be > 0.")
    if jitter < 0:
        raise ValueError("jitter must be >= 0.")

    if jitter == 0.0:
        return torch.full((n,), float(snr_value), device=device)

    lo = max(1e-6, snr_value - jitter)
    hi = snr_value + jitter
    return torch.empty((n,), device=device).uniform_(lo, hi)


# -----------------------------------------------------------------------------
# Policy application (what the pipeline will call)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseConfig:
    noise_type: NoiseType = "gaussian"
    strategy: NoiseStrategy = "random"

    snr_min: float = 2.0
    snr_max: float = 80.0
    n_levels: int = 8  # multilevel only

    snr_fixed: float = 30.0        # fixed only
    snr_fixed_jitter: float = 0.0  # fixed only

    device: str = "cpu"


def apply_noise_policy(
    x_noisefree: torch.Tensor,
    cfg: NoiseConfig,
    base_repeat: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a noise policy to noisefree signals.

    Returns
    -------
    x_noisy : torch.Tensor
        Noisy signals.
    snr_values : torch.Tensor
        SNR values used (shape (N,) for random/fixed; (N*n_levels,) for multilevel)
    """
    if x_noisefree.ndim != 2:
        raise ValueError("Expected x_noisefree with shape (N, G).")

    N, _G = x_noisefree.shape

    if cfg.strategy == "random":
        snr = sample_snr_random(N, cfg.snr_min, cfg.snr_max, device=cfg.device)
        x_noisy = add_noise_batch(x_noisefree, snr, noise_type=cfg.noise_type)
        return x_noisy, snr

    if cfg.strategy == "multilevel":
        snr = sample_snr_multilevel(N, cfg.snr_min, cfg.snr_max, cfg.n_levels, device=cfg.device)
        if base_repeat:
            x_rep = x_noisefree.repeat(cfg.n_levels, 1)  # (N*n_levels, G)
        else:
            x_rep = x_noisefree
            if x_rep.shape[0] != snr.shape[0]:
                raise ValueError("x_noisefree must already be repeated if base_repeat=False.")
        x_noisy = add_noise_batch(x_rep, snr, noise_type=cfg.noise_type)
        return x_noisy, snr

    if cfg.strategy == "fixed":
        snr = sample_snr_fixed(
            N,
            snr_value=cfg.snr_fixed,
            jitter=cfg.snr_fixed_jitter,
            device=cfg.device,
        )
        x_noisy = add_noise_batch(x_noisefree, snr, noise_type=cfg.noise_type)
        return x_noisy, snr

    raise ValueError(f"Unknown strategy: {cfg.strategy}")