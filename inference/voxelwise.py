"""
Voxelwise inference utilities for SBI-based dMRI models.

This module provides vectorised inference on masked volumetric data using
SBI density estimators. It is designed for efficiency (batch sampling)
and for diffusion MRI models with voxelwise independent parameters.

Important assumptions
---------------------
- The density estimator `estimator` implements:
      estimator.sample(n_samples, x)
  where `x` has shape (N, signal_dim).

- Each voxel is treated independently.
- Mask must match data spatial dimensions (without signal dimension).
- Model-specific post-processing (e.g. fibre sorting) is explicit.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional

from utils.sort_f import sort_f  # model-specific, keep explicit


def flatten_masked_data(
    data: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Flatten masked volumetric data for vectorised inference.

    Parameters
    ----------
    data
        Array of shape (X, Y, Z, signal_dim) or compatible.
    mask
        Binary mask of shape (X, Y, Z).

    Returns
    -------
    data_subset
        Tensor of shape (N_valid, signal_dim).
    indices
        Indices of valid voxels in the flattened array.
    original_shape
        Original spatial shape (X, Y, Z).
    """
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)

    data = data.float()
    mask = mask.bool()

    # Replace NaNs early
    data = torch.nan_to_num(data)

    spatial_shape = data.shape[:-1]
    signal_dim = data.shape[-1]

    data_flat = data.reshape(-1, signal_dim)
    mask_flat = mask.reshape(-1)

    indices = mask_flat.nonzero(as_tuple=False).squeeze()
    data_subset = data_flat[indices]

    return data_subset, indices, spatial_shape


def infer_voxelwise(
    estimator,
    data: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    n_samples: int,
    *,
    sort_fibres: bool = True,
) -> torch.Tensor:
    """
    Perform voxelwise SBI inference on masked volumetric data.

    Parameters
    ----------
    estimator
        Trained SBI density estimator (e.g. SNPE-C density estimator).
    data
        Observed data of shape (X, Y, Z, signal_dim).
    mask
        Binary mask of shape (X, Y, Z).
    n_samples
        Number of posterior samples per voxel.
    sort_fibres
        Whether to sort fibres by volume fraction (model-specific).

    Returns
    -------
    samples
        Tensor of shape (X, Y, Z, n_samples, n_params).
    """
    # ------------------------------------------------------------
    # Flatten masked data
    # ------------------------------------------------------------
    data_subset, indices, spatial_shape = flatten_masked_data(data, mask)

    # ------------------------------------------------------------
    # SBI sampling (vectorised)
    # ------------------------------------------------------------
    # Output shape: (N_valid, n_samples, n_params)
    samples_subset = estimator.sample(n_samples, data_subset)

    # ------------------------------------------------------------
    # Model-specific post-processing
    # ------------------------------------------------------------
    if sort_fibres:
        # Sort fibres using mean parameters
        idx = sort_f(torch.mean(samples_subset, dim=1))
        samples_subset = samples_subset[:, :, idx]

    # ------------------------------------------------------------
    # Reshape back to volume
    # ------------------------------------------------------------
    n_params = samples_subset.shape[-1]
    output_shape = spatial_shape + (n_samples, n_params)

    samples = torch.zeros(
        int(np.prod(spatial_shape)),
        n_samples,
        n_params,
        dtype=samples_subset.dtype,
    )

    samples[indices] = samples_subset
    samples = samples.reshape(output_shape)

    return samples