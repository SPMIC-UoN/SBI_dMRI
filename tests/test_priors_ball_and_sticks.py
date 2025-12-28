import torch
import numpy as np

from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors, omega_dim


def test_ball_and_sticks_prior_shape_model2_nfib3():
    cfg = BallAndSticksPriorConfig(nfib=3, modelnum=2, hemisphere=True, device="cpu")
    prior = build_ball_and_sticks_priors(cfg)

    n = 128
    s = prior.sample((n,))
    assert s.shape == (n, omega_dim(3, 2))


def test_ball_and_sticks_angles_ranges_hemisphere():
    cfg = BallAndSticksPriorConfig(nfib=2, modelnum=2, hemisphere=True, device="cpu")
    prior = build_ball_and_sticks_priors(cfg)
    s = prior.sample((512,))

    # layout: [d, f1, th1, phi1, f2, th2, phi2, d_std]
    th1 = s[:, 2]
    phi1 = s[:, 3]

    assert torch.all(th1 >= 0.0) and torch.all(th1 <= torch.pi)
    assert torch.all(phi1 >= 0.0) and torch.all(phi1 <= torch.pi)


def test_ball_and_sticks_angles_ranges_sphere():
    cfg = BallAndSticksPriorConfig(nfib=2, modelnum=2, hemisphere=False, device="cpu")
    prior = build_ball_and_sticks_priors(cfg)
    s = prior.sample((512,))

    phi1 = s[:, 3]
    assert torch.all(phi1 >= 0.0) and torch.all(phi1 <= 2.0 * torch.pi)


def test_ball_and_sticks_positive_d_model2():
    cfg = BallAndSticksPriorConfig(nfib=1, modelnum=2, hemisphere=True, device="cpu")
    prior = build_ball_and_sticks_priors(cfg)
    s = prior.sample((256,))

    d = s[:, 0]
    d_std = s[:, -1]
    assert torch.all(d > 0.0)
    assert torch.all(d_std >= 0.0)