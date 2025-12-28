"""
Post-processing utilities for the Ball-and-Sticks model.

This module contains ONLY model-specific logic:
- Parameter layout assumptions for ball-and-sticks (Ω vector structure)
- Fibre sorting by volume fraction
- Conversion of spherical angles (theta, phi) to Cartesian unit vectors ("dyads")
- Uncertainty quantification via dyadic-tensor eigendecomposition (BedpostX-like)

It intentionally does NOT:
- Load NIfTI files
- Apply masks or reshape volumes (see inference/voxelwise.py)
- Run SBI training or sampling

Conventions
-----------
Angles follow the common spherical convention:
- theta ∈ [0, π]   polar angle (from +z axis)
- phi   ∈ [0, 2π)  azimuth (in xy-plane)

Shapes
------
Posterior samples are expected as:
- samples: (..., n_samples, n_params)  (torch.Tensor or np.ndarray)
Where "..." is typically (X, Y, Z) for volumes or (N_vox,) if flattened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Literal

import numpy as np
import torch

try:
    import scipy.linalg as la
except ImportError as e:
    raise ImportError(
        "models.ball_and_sticks.postprocess requires scipy for dyads/dispersion "
        "(pip install scipy)."
    ) from e


ArrayLike = Union[np.ndarray, torch.Tensor]


# ======================================================================
# Layout (parameter indexing)
# ======================================================================

@dataclass(frozen=True)
class BallAndSticksLayout:
    """
    Parameter layout for Ball-and-Sticks samples.

    Expected Ω ordering:
        d,
        [f1, th1, ph1, f2, th2, ph2, ..., fN, thN, phN],
        (optional) d_std  if modelnum==2,
        (optional) SNR    if include_snr==True

    Notes
    -----
    - This matches your legacy conventions.
    - The class makes indexing explicit, testable, and less error-prone.
    """
    nfib: int
    modelnum: int = 1
    include_snr: bool = False

    @property
    def has_d_std(self) -> bool:
        return self.modelnum == 2

    @property
    def n_params(self) -> int:
        n = 1 + 3 * self.nfib  # d + (f,th,ph)*nfib
        if self.has_d_std:
            n += 1
        if self.include_snr:
            n += 1
        return n

    def idx_d(self) -> int:
        return 0

    def idx_f(self, i: int) -> int:
        """Index of f_i (1-based fibre index)."""
        if i < 1 or i > self.nfib:
            raise ValueError(f"Fibre index out of range: i={i}, nfib={self.nfib}")
        return 1 + 3 * (i - 1)

    def idx_theta(self, i: int) -> int:
        """Index of theta_i (1-based)."""
        return self.idx_f(i) + 1

    def idx_phi(self, i: int) -> int:
        """Index of phi_i (1-based)."""
        return self.idx_f(i) + 2

    def idx_d_std(self) -> Optional[int]:
        """Index of d_std, if present."""
        if not self.has_d_std:
            return None
        return 1 + 3 * self.nfib

    def idx_snr(self) -> Optional[int]:
        """Index of SNR, if present."""
        if not self.include_snr:
            return None
        return self.n_params - 1


# ======================================================================
# Helpers
# ======================================================================

def _to_torch(x: ArrayLike, device: Optional[str] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))
    if device is not None:
        t = t.to(device)
    return t


def _safe_unit_vector(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors safely to unit length."""
    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return v / norm


def sph_to_cart(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical angles to Cartesian unit vectors.

    Parameters
    ----------
    theta : torch.Tensor
        Polar angle, shape (...).
    phi : torch.Tensor
        Azimuth angle, shape (...).

    Returns
    -------
    v : torch.Tensor
        Cartesian unit vector, shape (..., 3).
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    v = torch.stack([x, y, z], dim=-1)
    return _safe_unit_vector(v)


# ======================================================================
# Fibre sorting
# ======================================================================

def sort_fibres_by_fraction(
    samples: ArrayLike,
    layout: BallAndSticksLayout,
    *,
    mode: Literal["mean"] = "mean",
) -> ArrayLike:
    """
    Sort fibre blocks (f,theta,phi) in descending order of fibre fraction f.

    Parameters
    ----------
    samples
        Shape (..., n_samples, n_params) or (..., n_params).
    layout
        Layout defining nfib and parameter indices.
    mode
        Sorting criterion. Currently only "mean" is supported:
        sort by mean f across posterior samples.

    Returns
    -------
    sorted_samples
        Same type as input (np.ndarray or torch.Tensor), with fibre blocks permuted.
    """
    if mode != "mean":
        raise ValueError("Only mode='mean' is currently supported.")

    is_numpy = isinstance(samples, np.ndarray)
    t = _to_torch(samples).float()

    if t.shape[-1] != layout.n_params:
        raise ValueError(f"Expected last dim {layout.n_params}, got {t.shape[-1]}")

    # Detect whether there's a posterior sample dimension:
    # - expected: (..., n_samples, n_params)
    # - or:       (..., n_params)
    has_sample_dim = t.ndim >= 2 and t.shape[-2] != layout.n_params

    if has_sample_dim:
        n_samp = t.shape[-2]
        f_means = torch.stack(
            [t[..., :, layout.idx_f(i)].mean(dim=-1) for i in range(1, layout.nfib + 1)],
            dim=-1,
        )
    else:
        n_samp = None
        f_means = torch.stack(
            [t[..., layout.idx_f(i)] for i in range(1, layout.nfib + 1)],
            dim=-1,
        )

    perm = torch.argsort(f_means, dim=-1, descending=True)  # (..., nfib)

    out = t.clone()

    f_idx = torch.tensor([layout.idx_f(i) for i in range(1, layout.nfib + 1)])
    th_idx = torch.tensor([layout.idx_theta(i) for i in range(1, layout.nfib + 1)])
    ph_idx = torch.tensor([layout.idx_phi(i) for i in range(1, layout.nfib + 1)])

    if has_sample_dim:
        # Gather expects index shape to match gathered dims
        perm_exp = perm.unsqueeze(-2).expand(*perm.shape[:-1], n_samp, perm.shape[-1])

        f_sorted = torch.gather(t[..., :, f_idx], dim=-1, index=perm_exp)
        th_sorted = torch.gather(t[..., :, th_idx], dim=-1, index=perm_exp)
        ph_sorted = torch.gather(t[..., :, ph_idx], dim=-1, index=perm_exp)

        for j in range(layout.nfib):
            out[..., :, layout.idx_f(j + 1)] = f_sorted[..., :, j]
            out[..., :, layout.idx_theta(j + 1)] = th_sorted[..., :, j]
            out[..., :, layout.idx_phi(j + 1)] = ph_sorted[..., :, j]
    else:
        f_sorted = torch.gather(t[..., f_idx], dim=-1, index=perm)
        th_sorted = torch.gather(t[..., th_idx], dim=-1, index=perm)
        ph_sorted = torch.gather(t[..., ph_idx], dim=-1, index=perm)

        for j in range(layout.nfib):
            out[..., layout.idx_f(j + 1)] = f_sorted[..., j]
            out[..., layout.idx_theta(j + 1)] = th_sorted[..., j]
            out[..., layout.idx_phi(j + 1)] = ph_sorted[..., j]

    return out.cpu().numpy() if is_numpy else out


# ======================================================================
# Dyads & dispersion (dyadic tensor method: your code)
# ======================================================================

def make_dyads(
    theta_samples: np.ndarray,
    phi_samples: np.ndarray,
    percentile: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute principal dyad direction and dispersion from spherical samples
    using the dyadic tensor eigen-decomposition.

    This is the method you provided (BedpostX-like uncertainty).

    Parameters
    ----------
    theta_samples : (S,) array
        Polar angles in radians.
    phi_samples : (S,) array
        Azimuth angles in radians.
    percentile : float, optional
        If provided, returns the cone angle (in degrees) containing this percentile
        of samples around the principal eigenvector.

    Returns
    -------
    v1 : (3,) array
        Principal eigenvector (mean fibre orientation).
    disp : float
        - If percentile is None: 1 - max(|eigenvalues|)
        - If percentile is set:  percentile cone angle in degrees
    """
    theta_samples = np.asarray(theta_samples).reshape(-1)
    phi_samples = np.asarray(phi_samples).reshape(-1)

    v = np.array(
        [
            np.sin(theta_samples) * np.cos(phi_samples),
            np.sin(theta_samples) * np.sin(phi_samples),
            np.cos(theta_samples),
        ]
    )

    dyadic_tensor = (v @ v.T) / len(theta_samples)

    L, E = la.eig(dyadic_tensor)  # L=eigenvalues, E=eigenvectors (columns)
    ind = np.argsort(-L.real, kind="quicksort")
    v1 = E[:, ind[0]].real

    disp = 1.0 - np.max(np.abs(L.real))

    if percentile is not None:
        angles = np.arccos(np.clip(v.T @ v1, -1.0, 1.0)) * (180.0 / np.pi)
        disp = float(np.percentile(angles, percentile))

    return v1, float(disp)


def make_dyads_cart(
    Vsamples: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute principal dyad direction and dispersion from Cartesian vector samples
    using the dyadic tensor eigen-decomposition.

    Parameters
    ----------
    Vsamples : (S, 3) array
        Cartesian unit vectors.

    Returns
    -------
    v1 : (3,) array
        Principal eigenvector.
    disp : float
        Dispersion = 1 - max(|eigenvalues|)
    """
    Vsamples = np.asarray(Vsamples)
    if Vsamples.ndim != 2 or Vsamples.shape[1] != 3:
        raise ValueError("Vsamples must have shape (S, 3).")

    N = Vsamples.shape[0]
    dyadic_tensor = np.zeros((3, 3), dtype=float)

    for l in range(N):
        v = Vsamples[l, :].reshape(3, 1)
        dyadic_tensor += v @ v.T

    dyadic_tensor /= N

    evals, evecs = la.eig(dyadic_tensor)
    ind = np.argsort(-evals.real, kind="quicksort")
    v1 = evecs[:, ind[0]].real
    disp = 1.0 - np.max(np.abs(evals.real))

    return v1, float(disp)


def dyads_and_dispersion_from_samples(
    samples: ArrayLike,
    layout: BallAndSticksLayout,
    *,
    fibre_index: int = 1,
    percentile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dyads and dispersion maps from posterior samples for one fibre.

    Parameters
    ----------
    samples
        Shape (..., S, P), where:
          ... = spatial dims (X,Y,Z) or flattened voxels (N,)
          S = number of posterior samples
          P = number of parameters
    layout
        Parameter layout.
    fibre_index
        1-based fibre index.
    percentile
        If provided, dispersion is the percentile cone angle (degrees),
        else uses 1 - max(|eigenvalues|).

    Returns
    -------
    v1 : (..., 3) array
        Principal dyad direction per voxel.
    disp : (...) array
        Dispersion per voxel.
    """
    if isinstance(samples, torch.Tensor):
        arr = samples.detach().cpu().numpy()
    else:
        arr = np.asarray(samples)

    if arr.ndim < 3:
        raise ValueError("samples must have shape (..., S, P).")

    S = arr.shape[-2]
    P = arr.shape[-1]
    if P != layout.n_params:
        raise ValueError(f"Expected P={layout.n_params}, got P={P}.")

    th_idx = layout.idx_theta(fibre_index)
    ph_idx = layout.idx_phi(fibre_index)

    thetas = arr[..., :, th_idx]  # (..., S)
    phis = arr[..., :, ph_idx]    # (..., S)

    lead_shape = thetas.shape[:-1]  # (...) excluding S
    n_vox = int(np.prod(lead_shape)) if len(lead_shape) else 1

    thetas_flat = thetas.reshape(n_vox, S)
    phis_flat = phis.reshape(n_vox, S)

    v1_out = np.zeros((n_vox, 3), dtype=float)
    disp_out = np.zeros((n_vox,), dtype=float)

    for i in range(n_vox):
        v1, disp = make_dyads(thetas_flat[i], phis_flat[i], percentile=percentile)
        v1_out[i] = v1
        disp_out[i] = disp

    return v1_out.reshape(lead_shape + (3,)), disp_out.reshape(lead_shape)


# ======================================================================
# Optional: scalar parameter summary maps
# ======================================================================

def extract_parameter_maps(
    samples: ArrayLike,
    layout: BallAndSticksLayout,
    *,
    summary: Literal["mean", "median"] = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Extract summary maps for scalar parameters (d, f_i, optional d_std, optional SNR).

    Parameters
    ----------
    samples
        Posterior samples, shape (..., S, P).
    layout
        Parameter layout.
    summary
        Summary statistic across posterior samples.

    Returns
    -------
    maps
        Dict of tensors keyed by parameter name.
        Each tensor has shape (...) (spatial dims only).
    """
    t = _to_torch(samples).float()

    if t.ndim < 3:
        raise ValueError("Expected samples with shape (..., S, P).")

    if t.shape[-1] != layout.n_params:
        raise ValueError(f"Expected last dim {layout.n_params}, got {t.shape[-1]}")

    if summary == "mean":
        reduce = lambda x: torch.mean(x, dim=-1)
    elif summary == "median":
        reduce = lambda x: torch.median(x, dim=-1).values
    else:
        raise ValueError("summary must be 'mean' or 'median'.")

    maps: Dict[str, torch.Tensor] = {}

    # d
    maps["d"] = reduce(t[..., :, layout.idx_d()])

    # fractions
    for i in range(1, layout.nfib + 1):
        maps[f"f{i}"] = reduce(t[..., :, layout.idx_f(i)])

    # d_std (model 2)
    if layout.has_d_std:
        idx = layout.idx_d_std()
        assert idx is not None
        maps["d_std"] = reduce(t[..., :, idx])

    # SNR (optional)
    if layout.include_snr:
        idx = layout.idx_snr()
        assert idx is not None
        maps["snr"] = reduce(t[..., :, idx])

    return maps