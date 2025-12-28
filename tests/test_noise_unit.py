# pytest -q tests/test_noise_unit.py

"""
tests/test_noise_unit.py

Unit tests for noise primitives in tools/noise.py.

Goals
-----
- Validate basic statistical behavior:
  * Gaussian: variance increases as SNR decreases (sigma=1/SNR)
  * Rician: outputs are non-negative and show upward bias at low SNR
- Save quick visual sanity plots to test_plots/ (headless-friendly)

Run
---
pytest -q tests/test_noise_unit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Headless-friendly plotting for pytest/HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.noise import add_noise  # noqa: E402


PLOT_DIR = Path("test_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_gaussian_noise_variance_monotonic_with_snr():
    """
    For Gaussian noise with sigma=1/SNR, the output variance around the noisefree
    signal should be higher when SNR is lower.

    We test this with a constant signal.
    """
    _set_seed(0)

    G = 200   # signal length
    N = 2000  # repeated noise draws (moderate for speed)
    x0 = torch.full((G,), 0.6, dtype=torch.float32)

    snr_high = 80.0
    snr_low = 5.0

    res_high = []
    res_low = []
    for _ in range(N):
        y_high = add_noise(x0, snr_high, noise_type="gaussian")
        y_low = add_noise(x0, snr_low, noise_type="gaussian")
        res_high.append((y_high - x0).numpy())
        res_low.append((y_low - x0).numpy())

    res_high = np.concatenate(res_high)
    res_low = np.concatenate(res_low)

    var_high = np.var(res_high)
    var_low = np.var(res_low)

    # Expect much larger variance at low SNR
    assert var_low > var_high * 50.0

    plt.figure()
    plt.hist(res_high, bins=120, density=True, alpha=0.6, label=f"SNR={snr_high}")
    plt.hist(res_low, bins=120, density=True, alpha=0.6, label=f"SNR={snr_low}")
    plt.title("Gaussian noise residuals (y - x0)")
    plt.xlabel("residual")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "noise_gaussian_residuals_hist.png", dpi=150)
    plt.close()


def test_rician_noise_is_nonnegative_and_biased_at_low_snr():
    """
    Rician magnitude noise should always be >= 0 and introduces an upward bias,
    especially at low SNR.

    We'll compare the mean of noisy observations with the noisefree mean.
    """
    _set_seed(1)

    G = 200
    N = 2000
    x0 = torch.full((G,), 0.1, dtype=torch.float32)  # low signal enhances bias visibility

    snr_low = 5.0
    snr_high = 80.0

    ys_low = []
    ys_high = []
    for _ in range(N):
        ys_low.append(add_noise(x0, snr_low, noise_type="rician").numpy())
        ys_high.append(add_noise(x0, snr_high, noise_type="rician").numpy())

    ys_low = np.concatenate(ys_low)
    ys_high = np.concatenate(ys_high)

    # Non-negativity
    assert np.min(ys_low) >= 0.0
    assert np.min(ys_high) >= 0.0

    mean_x0 = float(x0.mean().item())
    mean_low = float(np.mean(ys_low))
    mean_high = float(np.mean(ys_high))

    # Bias: at low SNR should be notably above noisefree mean
    assert mean_low > mean_x0

    # At high SNR, bias is smaller than at low SNR
    assert abs(mean_high - mean_x0) < abs(mean_low - mean_x0)

    plt.figure()
    plt.hist(ys_high, bins=120, density=True, alpha=0.6, label=f"Rician SNR={snr_high}")
    plt.hist(ys_low, bins=120, density=True, alpha=0.6, label=f"Rician SNR={snr_low}")
    plt.axvline(mean_x0, linestyle="--", label="noisefree mean")
    plt.title("Rician noisy samples distribution")
    plt.xlabel("signal value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "noise_rician_hist.png", dpi=150)
    plt.close()


def test_noise_preserves_shape():
    """add_noise() should preserve the input tensor shape."""
    _set_seed(2)

    x = torch.rand((7, 13), dtype=torch.float32)
    y = add_noise(x, 20.0, noise_type="gaussian")
    z = add_noise(x, 20.0, noise_type="rician")

    assert y.shape == x.shape
    assert z.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(z).all()