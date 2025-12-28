# pytest -q tests/test_simulator_ball_and_sticks_unit.py


"""
tests/test_simulator_ball_and_sticks_unit.py

UNIT TESTS for the Ball-and-Sticks attenuation simulator.

Scope
-----
These tests validate the *numerical correctness and internal logic* of the
BallAndSticksAttenuation forward model **in isolation**, using hand-crafted,
deterministic parameter vectors.

What these tests are meant to catch
-----------------------------------
- Shape regressions (wrong output dimensions)
- NaNs / infs introduced by refactors
- Physically impossible attenuation values
- Bugs in modelnum=2 vs modelnum=1 branching logic
- Silent changes in parameter ordering or interpretation

What these tests deliberately do NOT test
------------------------------------------
- Priors
- SBI integration
- Training / inference
- Statistical properties of parameters

If any of these tests fail, the simulator itself is broken.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from utils_plotting import _save_example_signals
from models.ball_and_sticks.simulator import GradientTable, BallAndSticksAttenuation


def _random_unit_bvecs(G: int, seed: int = 0) -> np.ndarray:
    """
    Generate G random unit vectors on the sphere.

    Used to construct a synthetic gradient table that is:
    - deterministic (seeded)
    - physically valid (||bvec|| = 1)
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, G)).astype(np.float32)
    v /= np.linalg.norm(v, axis=0, keepdims=True) + 1e-12
    return v


def _make_gtab(G: int = 60, bval: float = 1000.0) -> GradientTable:
    """
    Construct a minimal GradientTable suitable for attenuation models.

    Notes
    -----
    - Single b-value (no b0)
    - Random directions
    - This avoids any dependency on real acquisition files
    """
    bvals = np.full((G,), bval, dtype=np.float32)
    bvecs = _random_unit_bvecs(G)
    return GradientTable(
        bvals=torch.tensor(bvals, dtype=torch.float32),
        bvecs=torch.tensor(bvecs, dtype=torch.float32),
    )


def test_simulator_output_shape_and_finiteness_model2():
    """
    Basic smoke test for modelnum=2.

    Verifies that:
    - Output shape is (N, G)
    - All values are finite
    """
    gtab = _make_gtab(G=64, bval=1200.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    nfib = 3
    modelnum = 2
    P = sim.expected_param_dim(nfib=nfib, modelnum=modelnum)

    N = 16
    params = torch.zeros((N, P), dtype=torch.float32)

    # Core parameters
    params[:, 0] = 0.002   # d
    params[:, 1] = 0.4     # f1
    params[:, 4] = 0.2     # f2
    params[:, 7] = 0.1     # f3
    params[:, -1] = 0.001  # d_std

    # Orientations (arbitrary but valid)
    params[:, 2] = 0.7; params[:, 3] = 1.2
    params[:, 5] = 1.1; params[:, 6] = 2.0
    params[:, 8] = 2.2; params[:, 9] = 0.4

    y = sim(params, nfib=nfib, modelnum=modelnum)

    assert y.shape == (N, gtab.bvals.numel())
    assert torch.isfinite(y).all()


def test_attenuation_range_is_reasonable():
    """
    Physically motivated sanity check.

    Attenuation S/S0 should lie in approximately [0, 1] for
    reasonable parameter values.
    """
    gtab = _make_gtab(G=50, bval=1500.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    nfib = 2
    modelnum = 2
    P = sim.expected_param_dim(nfib=nfib, modelnum=modelnum)

    params = torch.zeros((8, P), dtype=torch.float32)
    params[:, 0] = 0.002
    params[:, 1] = 0.5
    params[:, 4] = 0.2
    params[:, 2] = 1.0; params[:, 3] = 0.5
    params[:, 5] = 2.0; params[:, 6] = 1.3
    params[:, -1] = 0.001

    y = sim(params, nfib=nfib, modelnum=modelnum)

    assert torch.max(y) <= 1.0 + 1e-6
    assert torch.min(y) >= 0.0 - 1e-8


def test_model2_small_dstd_matches_model1_branch():
    """
    Regression test for modelnum=2 → modelnum=1 fallback.

    When d_std → 0, modelnum=2 should behave identically
    to modelnum=1.
    """
    gtab = _make_gtab(G=40, bval=1000.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    nfib = 2

    # Model 1 parameters
    P1 = sim.expected_param_dim(nfib=nfib, modelnum=1)
    p1 = torch.zeros((10, P1), dtype=torch.float32)
    p1[:, 0] = 0.002
    p1[:, 1] = 0.6
    p1[:, 4] = 0.2
    p1[:, 2] = 0.9; p1[:, 3] = 0.3
    p1[:, 5] = 1.7; p1[:, 6] = 2.2

    y1 = sim(p1, nfib=nfib, modelnum=1)

    # Same params + tiny d_std
    P2 = sim.expected_param_dim(nfib=nfib, modelnum=2)
    p2 = torch.zeros((10, P2), dtype=torch.float32)
    p2[:, :P1] = p1
    p2[:, -1] = 1e-8

    y2 = sim(p2, nfib=nfib, modelnum=2)

    assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-7)


def test_simulator_output_shape_and_finiteness_model2():
    gtab = _make_gtab(G=64, bval=1200.0)
    sim = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    nfib = 3
    modelnum = 2
    P = sim.expected_param_dim(nfib=nfib, modelnum=modelnum)

    N = 16
    params = torch.zeros((N, P), dtype=torch.float32)
    params[:, 0] = 0.002
    params[:, 1] = 0.4
    params[:, 4] = 0.2
    params[:, 7] = 0.1
    params[:, -1] = 0.001

    params[:, 2] = 0.7; params[:, 3] = 1.2
    params[:, 5] = 1.1; params[:, 6] = 2.0
    params[:, 8] = 2.2; params[:, 9] = 0.4

    y = sim(params, nfib=nfib, modelnum=modelnum)

    assert y.shape == (N, gtab.bvals.numel())
    assert torch.isfinite(y).all()

    _save_example_signals(
        y,
        title="Ball-and-Sticks simulator (unit test, model 2)",
        filename="simulator_unit_model2_examples.png",
    )
