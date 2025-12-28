"""
models/ball_and_sticks/export_params.py

Export Ball-and-Sticks parameter samples/maps to per-parameter NIfTI files.

Accepts theta shaped (..., P), e.g.:
- (X, Y, Z, P)  parameter maps
- (N, P)        flattened voxels (will be exported as (N,1,1) unless you reshape first)

Requires:
- ref_nifti to preserve affine/header consistently (recommended).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from tools.io_nifti import export_any_to_nifti, to_numpy
from models.ball_and_sticks.postprocess import BallAndSticksLayout


ArrayLike = Union[np.ndarray, torch.Tensor]


def split_theta_to_param_arrays(
    theta: ArrayLike,
    layout: BallAndSticksLayout,
    *,
    squeeze_last: bool = True,
) -> Dict[str, ArrayLike]:
    """
    Split theta (..., P) into a dict of parameter arrays (...).

    Returns keys:
      d, f1..fN, theta1..thetaN, phi1..phiN, (optional) d_std, (optional) snr
    """
    t = theta if isinstance(theta, torch.Tensor) else torch.as_tensor(np.asarray(theta))
    if t.ndim < 1:
        raise ValueError("theta must have at least 1 dimension.")
    if t.shape[-1] != layout.n_params:
        raise ValueError(f"Expected theta last dim P={layout.n_params}, got {t.shape[-1]}")

    out: Dict[str, ArrayLike] = {}

    def _sl(name: str, idx: int):
        arr = t[..., idx]
        if squeeze_last:
            # arr is already (...), no trailing singleton dim
            pass
        out[name] = arr

    _sl("d", layout.idx_d())

    for i in range(1, layout.nfib + 1):
        _sl(f"f{i}", layout.idx_f(i))
        _sl(f"theta{i}", layout.idx_theta(i))
        _sl(f"phi{i}", layout.idx_phi(i))

    if layout.has_d_std:
        idx = layout.idx_d_std()
        assert idx is not None
        _sl("d_std", idx)

    if layout.include_snr:
        idx = layout.idx_snr()
        assert idx is not None
        _sl("snr", idx)

    return out


def export_theta_params_to_nifti(
    theta: ArrayLike,
    out_dir: Union[str, Path],
    *,
    nfib: int,
    modelnum: int,
    include_snr: bool = False,
    ref_nifti: Optional[Union[str, Path]] = None,
    dtype: Optional[np.dtype] = np.float32,
    prefix: str = "",
    overwrite: bool = True,
) -> Dict[str, Path]:
    """
    Export each parameter of theta (..., P) as an independent NIfTI in out_dir.

    Parameters
    ----------
    theta
        Array-like (..., P).
    out_dir
        Output folder.
    nfib, modelnum, include_snr
        Layout definition.
    ref_nifti
        Reference NIfTI for affine/header (strongly recommended).
    dtype
        Storage dtype (np.float32 recommended).
    prefix
        Optional filename prefix (e.g. "gt_" or "pred_").
    overwrite
        Overwrite existing files.

    Returns
    -------
    Dict[str, Path]
        Mapping param_name -> written file path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ref_nifti is None:
        raise ValueError("ref_nifti is required to export per-parameter NIfTIs reliably.")

    layout = BallAndSticksLayout(nfib=nfib, modelnum=modelnum, include_snr=include_snr)

    params = split_theta_to_param_arrays(theta, layout)

    written: Dict[str, Path] = {}
    for name, arr in params.items():
        fname = f"{prefix}{name}.nii.gz"
        out_path = out_dir / fname
        # export_any_to_nifti handles torch/numpy and preserves dims
        export_any_to_nifti(arr, out_path, ref_nifti=ref_nifti, dtype=dtype, overwrite=overwrite)
        written[name] = out_path

    # also write a tiny index map for sanity/debug
    idx_txt = []
    idx_txt.append(f"layout: nfib={nfib}, modelnum={modelnum}, include_snr={include_snr}")
    idx_txt.append(f"P={layout.n_params}")
    idx_txt.append(f"d: {layout.idx_d()}")
    for i in range(1, nfib + 1):
        idx_txt.append(f"f{i}: {layout.idx_f(i)}")
        idx_txt.append(f"theta{i}: {layout.idx_theta(i)}")
        idx_txt.append(f"phi{i}: {layout.idx_phi(i)}")
    if layout.has_d_std:
        idx_txt.append(f"d_std: {layout.idx_d_std()}")
    if include_snr:
        idx_txt.append(f"snr: {layout.idx_snr()}")

    (out_dir / f"{prefix}theta_param_indices.txt").write_text("\n".join(idx_txt) + "\n")

    return written