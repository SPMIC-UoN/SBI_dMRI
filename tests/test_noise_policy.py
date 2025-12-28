# pytest -q tests/test_noise_policy.py

"""
tests/test_noise_policy.py

Tests for the higher-level noise policy API in tools/noise.py:
- apply_noise_policy()
- strategies: "random", "multilevel", "fixed"

Also saves example signal overlays to test_plots/ for quick inspection.

Run
---
pytest -q tests/test_noise_policy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Headless-friendly plotting for pytest/HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.noise import NoiseConfig, apply_noise_policy  # noqa: E402


PLOT_DIR = Path("test_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_noisefree_batch(N: int = 32, G: int = 60) -> torch.Tensor:
    """
    Make a simple synthetic noisefree batch (attenuation-like, in [0,1]).
    Not tied to any forward model; just for noise policy tests.
    """
    t = torch.linspace(0, 1, G).unsqueeze(0).repeat(N, 1)
    # gentle curve between ~0.2 and ~0.9
    x = 0.2 + 0.7 * torch.exp(-2.5 * t)
    return x.float()


def test_apply_noise_policy_random_shapes_and_range():
    _set_seed(0)
    x0 = _make_noisefree_batch(N=40, G=80)

    cfg = NoiseConfig(
        noise_type="gaussian",
        strategy="random",
        snr_min=3.0,
        snr_max=80.0,
        device="cpu",
    )

    x_noisy, snr = apply_noise_policy(x0, cfg)

    assert x_noisy.shape == x0.shape
    assert snr.shape == (x0.shape[0],)
    assert torch.all(snr >= cfg.snr_min)
    assert torch.all(snr <= cfg.snr_max)
    assert torch.isfinite(x_noisy).all()

    # Save a quick overlay plot for a few examples
    k = 5
    plt.figure()
    for i in range(k):
        plt.plot(x0[i].numpy(), alpha=0.8, label="noisefree" if i == 0 else None)
        plt.plot(x_noisy[i].numpy(), alpha=0.8, linestyle="--", label="noisy" if i == 0 else None)
    plt.title("Noise policy: random (Gaussian) — example overlays")
    plt.xlabel("gradient index")
    plt.ylabel("signal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "noise_policy_random_overlays.png", dpi=150)
    plt.close()


def test_apply_noise_policy_multilevel_shapes_and_repeat_behavior():
    _set_seed(1)
    N, G = 20, 60
    x0 = _make_noisefree_batch(N=N, G=G)

    cfg = NoiseConfig(
        noise_type="gaussian",
        strategy="multilevel",
        snr_min=2.0,
        snr_max=80.0,
        n_levels=8,
        device="cpu",
    )

    x_noisy, snr = apply_noise_policy(x0, cfg, base_repeat=True)

    assert x_noisy.shape == (N * cfg.n_levels, G)
    assert snr.shape == (N * cfg.n_levels,)
    assert torch.all(snr >= cfg.snr_min)
    assert torch.all(snr <= cfg.snr_max)
    assert torch.isfinite(x_noisy).all()

    # Verify repeat behavior (shape-level check)
    assert x0.repeat(cfg.n_levels, 1).shape == x_noisy.shape

    # Save a plot showing the same underlying sample at multiple noise levels
    idx_base = 0
    plt.figure()
    plt.plot(x0[idx_base].numpy(), linewidth=2.0, label="noisefree")

    # plot one sample from each level (same base signal, different noise)
    for level in range(cfg.n_levels):
        i = level * N + idx_base
        plt.plot(x_noisy[i].numpy(), alpha=0.8, label=f"level {level} (SNR~{snr[i].item():.1f})")

    plt.title("Noise policy: multilevel (Gaussian) — same base signal, multiple SNRs")
    plt.xlabel("gradient index")
    plt.ylabel("signal")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "noise_policy_multilevel_same_base.png", dpi=150)
    plt.close()


def test_apply_noise_policy_fixed_shapes_and_constant_snr():
    _set_seed(123)
    x0 = _make_noisefree_batch(N=30, G=70)

    cfg = NoiseConfig(
        noise_type="gaussian",
        strategy="fixed",
        snr_fixed=25.0,
        snr_fixed_jitter=0.0,
        device="cpu",
    )

    x_noisy, snr = apply_noise_policy(x0, cfg)

    assert x_noisy.shape == x0.shape
    assert snr.shape == (x0.shape[0],)
    assert torch.allclose(snr, torch.full_like(snr, 25.0))
    assert torch.isfinite(x_noisy).all()

    plt.figure()
    plt.plot(x0[0].numpy(), label="noisefree")
    plt.plot(x_noisy[0].numpy(), linestyle="--", label="noisy (fixed SNR)")
    plt.title("Noise policy: fixed (Gaussian) — example overlay")
    plt.xlabel("gradient index")
    plt.ylabel("signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "noise_policy_fixed_overlay.png", dpi=150)
    plt.close()


def test_apply_noise_policy_rician_nonnegativity():
    _set_seed(2)
    x0 = _make_noisefree_batch(N=25, G=50)

    cfg = NoiseConfig(
        noise_type="rician",
        strategy="random",
        snr_min=3.0,
        snr_max=30.0,
        device="cpu",
    )

    x_noisy, snr = apply_noise_policy(x0, cfg)

    assert x_noisy.shape == x0.shape
    assert snr.shape == (x0.shape[0],)
    assert torch.all(x_noisy >= 0.0)
    assert torch.isfinite(x_noisy).all()