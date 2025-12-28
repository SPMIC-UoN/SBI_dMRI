#!/usr/bin/env python3
"""
tools/io_nifti.py

IO helpers for:
- Loading arrays from .pt / .npy / .nii/.nii.gz
- Exporting numpy/torch arrays to NIfTI using a reference NIfTI for geometry

Important:
- NIfTI (via nibabel) typically does NOT support float16 in headers.
  So exports will automatically cast float16 -> float32 unless dtype is explicitly set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import nibabel as nib


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, torch.Tensor]


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_any_array(path_or_obj: Union[PathLike, ArrayLike]) -> ArrayLike:
    """
    Load an array from disk (pt/npy/nifti) or pass-through if already an array/tensor.

    Returns
    -------
    np.ndarray | torch.Tensor
        - For .pt/.pth: returns the torch object as saved (often torch.Tensor)
        - For .npy: returns np.ndarray
        - For .nii/.nii.gz: returns np.ndarray
    """
    if isinstance(path_or_obj, (np.ndarray, torch.Tensor)):
        return path_or_obj

    p = Path(path_or_obj)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suf = "".join(p.suffixes).lower()

    if suf.endswith(".pt") or suf.endswith(".pth"):
        return torch.load(str(p), map_location="cpu")

    if suf.endswith(".npy"):
        return np.load(str(p), allow_pickle=False)

    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        img = nib.load(str(p))
        return np.asanyarray(img.dataobj)  # forces load into memory

    raise ValueError(f"Unsupported file type for: {p} (suffixes={p.suffixes})")


# -----------------------------------------------------------------------------
# Conversion helpers (PUBLIC)
# -----------------------------------------------------------------------------
def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor or array-like to a NumPy array on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_torch(x: ArrayLike, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy/torch to torch.Tensor on requested device."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# dtype policy for NIfTI
# -----------------------------------------------------------------------------
def _choose_nifti_dtype(data: np.ndarray, requested: Optional[str]) -> np.dtype:
    """
    Decide a safe dtype for NIfTI export.

    Rules:
    - if requested is not None: try to use it (but still protect against float16)
    - else: use data.dtype, but cast float16 -> float32
    - for non-float types: default to float32 (safe for param maps)
    """
    if requested is not None:
        dt = np.dtype(requested)
    else:
        dt = data.dtype

    # NIfTI/analyze header often rejects float16
    if dt == np.float16:
        dt = np.float32

    # Most maps should be float
    if not np.issubdtype(dt, np.floating):
        dt = np.float32

    return dt


# -----------------------------------------------------------------------------
# Exporting
# -----------------------------------------------------------------------------
def export_any_to_nifti(
    arr: ArrayLike,
    out_path: PathLike,
    *,
    ref_nifti: PathLike,
    dtype: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """
    Export an array/tensor to NIfTI using a reference NIfTI for affine + header.

    Notes
    -----
    - If arr is float16, we cast to float32 automatically (unless dtype explicitly says otherwise,
      but float16 will still be coerced to float32 due to NIfTI header limitations).
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {out_path}")

    ref_nifti = Path(ref_nifti)
    if not ref_nifti.exists():
        raise FileNotFoundError(f"Reference NIfTI not found: {ref_nifti}")

    ref_img = nib.load(str(ref_nifti))
    affine = ref_img.affine
    header = ref_img.header.copy()

    data = to_numpy(arr)
    out_dtype = _choose_nifti_dtype(data, dtype)

    data = data.astype(out_dtype, copy=False)
    header.set_data_dtype(out_dtype)

    img = nib.Nifti1Image(data, affine=affine, header=header)
    nib.save(img, str(out_path))


def export_theta_params_to_nifti(
    theta: ArrayLike,
    out_dir: PathLike,
    *,
    ref_nifti: PathLike,
    param_names: Optional[list[str]] = None,
    dtype: str = "float32",
    overwrite: bool = True,
) -> None:
    """
    Export theta with shape (..., P) as one NIfTI per parameter (last dim split).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = to_numpy(theta)
    if t.ndim < 2:
        raise ValueError(f"theta must have at least 2 dims (...,P). Got shape {t.shape}")

    P = t.shape[-1]
    if param_names is not None and len(param_names) != P:
        raise ValueError(f"param_names length {len(param_names)} != P={P}")

    if param_names is None:
        param_names = [f"param_{i:02d}" for i in range(P)]

    for i, name in enumerate(param_names):
        vol = t[..., i]
        export_any_to_nifti(
            vol,
            out_dir / f"{name}.nii.gz",
            ref_nifti=ref_nifti,
            dtype=dtype,
            overwrite=overwrite,
        )