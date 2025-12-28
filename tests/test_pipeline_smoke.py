import numpy as np
import torch

from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE_C

from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors, omega_dim
from models.ball_and_sticks.simulator import GradientTable, BallAndSticksAttenuation


def _random_unit_bvecs(G: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, G)).astype(np.float32)
    v /= np.linalg.norm(v, axis=0, keepdims=True) + 1e-12
    return v


def _make_gtab(G: int = 30, bval: float = 1200.0) -> GradientTable:
    # Attenuation model expects nob0s -> bvals must be >= 50-ish
    bvals = np.full((G,), bval, dtype=np.float32)
    bvecs = _random_unit_bvecs(G, seed=0)
    return GradientTable(
        bvals=torch.tensor(bvals, dtype=torch.float32),
        bvecs=torch.tensor(bvecs, dtype=torch.float32),
    )


def test_smoke_prior_simulate_train_infer_ball_and_sticks_model2_cpu():
    """
    End-to-end smoke test (fast):
      prior -> simulate_for_sbi -> SNPE_C train -> posterior sample

    This should run in a few seconds on CPU and catch:
    - broken priors (shape/ranges)
    - simulator signature / output shape mismatch
    - SBI wiring issues
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Small, fast settings
    nfib = 2
    modelnum = 2
    n_train = 256
    n_post = 8
    G = 24

    # Build prior
    prior_cfg = BallAndSticksPriorConfig(
        nfib=nfib,
        modelnum=modelnum,
        hemisphere=True,  # your default convention
        use_gamma_d=False,
        use_gamma_d_std=False,
        device="cpu",
    )
    prior = build_ball_and_sticks_priors(prior_cfg)

    # Build simulator (as a callable for sbi)
    gtab = _make_gtab(G=G, bval=1200.0)
    sim_impl = BallAndSticksAttenuation(gtab=gtab, device="cpu")

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        # theta is (P,) or (N,P); return (N,G)
        return sim_impl(theta, nfib=nfib, modelnum=modelnum)

    # Prepare for SBI (wraps shapes, ensures consistent batch behaviour)
    simulator_sbi, prior_sbi = prepare_for_sbi(simulator, prior)

    # Simulate training data
    theta, x = simulate_for_sbi(
        simulator=simulator_sbi,
        proposal=prior_sbi,
        num_simulations=n_train,
        num_workers=1,
        show_progress_bar=False,
    )

    # Basic sanity checks
    assert theta.shape == (n_train, omega_dim(nfib, modelnum))
    assert x.shape == (n_train, G)
    assert torch.isfinite(x).all()

    # Train NPE (SNPE-C)
    trainer = SNPE_C(
        prior=prior_sbi,
        density_estimator="nsf",
        device="cpu",
        show_progress_bars=False,
        logging_level="WARNING",
    )

    _ = trainer.append_simulations(theta, x).train(
        training_batch_size=64,
        max_num_epochs=2,          # keep it tiny for smoke test
        stop_after_epochs=2,
        show_train_summary=False,
    )

    posterior = trainer.build_posterior()

    # Posterior sampling at one observation
    x0 = x[0]
    samples = posterior.sample((n_post,), x=x0)

    assert samples.shape == (n_post, omega_dim(nfib, modelnum))
    assert torch.isfinite(samples).all()