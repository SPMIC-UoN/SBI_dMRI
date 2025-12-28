"""
tools/get_priors.py

Compatibility wrapper.

Prefer direct imports in new code:
    from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors
"""

from __future__ import annotations

from typing import Optional, Tuple

from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors


def get_priors_ball_and_sticks(
    nfib: int,
    modelnum: int,
    device: str = "cpu",
    *,
    include_snr: bool = False,
    snr_uniform: Tuple[float, float] = (2.0, 80.0),
    hemisphere: bool = True,
):
    """
    Build Ball-and-Sticks priors.

    include_snr=True appends SNR as the last parameter in theta, matching
    the paper convention: theta = [Ω, SNR].
    """
    cfg = BallAndSticksPriorConfig(
        nfib=nfib,
        modelnum=modelnum,
        device=device,
        hemisphere=hemisphere,
        include_snr=include_snr,
        snr_uniform=snr_uniform,
    )
    return build_ball_and_sticks_priors(cfg)