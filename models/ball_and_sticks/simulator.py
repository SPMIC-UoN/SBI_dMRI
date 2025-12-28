from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]


@dataclass(frozen=True)
class GradientTable:
    """
    Diffusion gradient table (bvals + bvecs).

    Notes
    -----
    - Expects bvals/bvecs in the common FSL-style text format:
        * bvals: 1D array of length G
        * bvecs: shape (3, G) or (G, 3)
    - Assumes b0 volumes have already been removed
      (attenuation model: S/S0).
    """
    bvals: torch.Tensor  # (G,)
    bvecs: torch.Tensor  # (3, G)

    @classmethod
    def read_bvals_bvecs(cls, bvals_path: PathLike, bvecs_path: PathLike, device: str = "cpu") -> "GradientTable":
        bvals = torch.tensor(np.genfromtxt(str(bvals_path), dtype=np.float32), device=device)
        bvecs = torch.tensor(np.genfromtxt(str(bvecs_path), dtype=np.float32), device=device)

        if bvals.ndim != 1:
            bvals = bvals.reshape(-1)

        # Accept (3,G) or (G,3) and store as (3,G)
        if bvecs.ndim != 2:
            raise ValueError("bvecs must be a 2D array.")
        if bvecs.shape[0] == 3:
            pass
        elif bvecs.shape[1] == 3:
            bvecs = bvecs.T
        else:
            raise ValueError("bvecs must have shape (3,G) or (G,3).")

        # Attenuation model requires b0 removed
        if torch.any(bvals < 50):
            raise ValueError(
                "Attenuation model requires bvals/bvecs without b0 volumes "
                "(bvals < 50 detected). Provide *nob0s* files."
            )

        if bvals.numel() != bvecs.shape[1]:
            raise ValueError(f"Mismatch: len(bvals)={bvals.numel()} but bvecs has G={bvecs.shape[1]} columns.")

        return cls(bvals=bvals, bvecs=bvecs)


class BallAndSticksAttenuation:
    """
    Ball-and-Sticks simulator for attenuation S/S0 (S0 fixed to 1).

    Ω layout:
      modelnum=1 (single-shell):
        [d, f1, th1, ph1, ..., fN, thN, phN]
        P = 1 + 3*nfib

      modelnum=2 (multi-shell / variable diffusivity):
        [d, f1, th1, ph1, ..., fN, thN, phN, d_std]
        P = 2 + 3*nfib  (d_std is last)

    Parameters
    ----------
    gtab : GradientTable
        bvals/bvecs for the acquisition protocol (no b0 volumes).
    device : str
        Torch device string, e.g. "cpu" or "cuda".
    """

    def __init__(self, gtab: GradientTable, device: str = "cpu"):
        self.gtab = gtab
        self.device = device

    @staticmethod
    def expected_param_dim(nfib: int, modelnum: int) -> int:
        if modelnum == 1:
            return 1 + 3 * nfib
        if modelnum == 2:
            return 2 + 3 * nfib
        raise ValueError("modelnum must be 1 or 2.")

    def __call__(self, params: torch.Tensor, nfib: int, modelnum: int = 2) -> torch.Tensor:
        """
        Simulate attenuation signal for batch of parameters.

        Parameters
        ----------
        params : torch.Tensor
            Shape (P,) or (N,P).
        nfib : int
            Number of fibres for this model.
        modelnum : int
            1 (single-shell) or 2 (multi-shell). Default: 2.

        Returns
        -------
        Sj : torch.Tensor
            Shape (N,G) attenuation signal.
        """
        if params.ndim == 1:
            params = params.unsqueeze(0)

        params = params.to(self.device).float()
        N, P = params.shape

        P_expected = self.expected_param_dim(nfib=nfib, modelnum=modelnum)
        if P != P_expected:
            raise ValueError(f"Parameter dimension mismatch: got P={P}, expected P={P_expected} for nfib={nfib}, modelnum={modelnum}.")

        b = self.gtab.bvals.to(self.device)         # (G,)
        g = self.gtab.bvecs.to(self.device).T       # (G,3)

        d = params[:, 0]                            # (N,)
        Sj = torch.zeros((N, b.numel()), dtype=params.dtype, device=self.device)
        sumf = torch.zeros((N,), dtype=params.dtype, device=self.device)

        def stick_attenuation(v: torch.Tensor, d_eff: torch.Tensor) -> torch.Tensor:
            dot = torch.matmul(v, g.T)              # (N,G)
            return torch.exp(-d_eff[:, None] * b[None, :] * dot.pow(2))

        # -------------------------
        # modelnum=1 (or degenerate)
        # -------------------------
        if modelnum == 1:
            for i in range(nfib):
                fi = params[:, 1 + 3 * i]
                th = params[:, 2 + 3 * i]
                ph = params[:, 3 + 3 * i]
                v = torch.stack(
                    [torch.sin(th) * torch.cos(ph),
                     torch.sin(th) * torch.sin(ph),
                     torch.cos(th)],
                    dim=-1,
                )  # (N,3)

                sumf += fi
                Sj += fi[:, None] * stick_attenuation(v, d)

            Sj += (1.0 - sumf)[:, None] * torch.exp(-b[None, :] * d[:, None])
            return Sj

        # -------------------------
        # modelnum=2
        # -------------------------
        d_std = params[:, -1]
        # If d_std is ~0, fall back to model 1
        if torch.all(d_std <= 1e-5):
            for i in range(nfib):
                fi = params[:, 1 + 3 * i]
                th = params[:, 2 + 3 * i]
                ph = params[:, 3 + 3 * i]
                v = torch.stack(
                    [torch.sin(th) * torch.cos(ph),
                     torch.sin(th) * torch.sin(ph),
                     torch.cos(th)],
                    dim=-1,
                )
                sumf += fi
                Sj += fi[:, None] * stick_attenuation(v, d)

            Sj += (1.0 - sumf)[:, None] * torch.exp(-b[None, :] * d[:, None])
            return Sj

        sig2 = d_std.pow(2)
        d_alpha = d.pow(2) / sig2  # (N,)

        for i in range(nfib):
            fi = params[:, 1 + 3 * i]
            th = params[:, 2 + 3 * i]
            ph = params[:, 3 + 3 * i]
            v = torch.stack(
                [torch.sin(th) * torch.cos(ph),
                 torch.sin(th) * torch.sin(ph),
                 torch.cos(th)],
                dim=-1,
            )  # (N,3)

            sumf += fi
            angp2 = torch.matmul(v, g.T).pow(2)  # (N,G)

            Sj += fi[:, None] * torch.exp(
                d_alpha[:, None] * torch.log(
                    d[:, None] / (d[:, None] + b[None, :] * angp2 * sig2[:, None])
                )
            )

        Sj += (1.0 - sumf)[:, None] * torch.exp(
            d_alpha[:, None] * torch.log(
                d[:, None] / (d[:, None] + b[None, :] * sig2[:, None])
            )
        )

        return Sj