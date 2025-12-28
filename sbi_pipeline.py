"""
sbi_pipeline.py

SBI dMRI pipeline (TRAINING-ONLY for now) driven by a Python config file.

Key changes
-----------
1) Per-run folder organization:
   results_root/model_name/run_YYYYMMDD_HHMMSS/
       - configfile.py
       - logs/
       - data/
       - models/

2) Dataset storage for 5-10M simulations:
   - Save as Torch tensors
   - Store x (and x_noisefree optionally) as float16 on disk
   - theta stored as float32 (recommended)
   - Use real compression (lzma by default) via tools/io_compressed.py

3) Noise is applied OUTSIDE the simulator:
   - simulate noisefree: (theta_omega, x0)
   - apply_noise_policy(x0) -> (x_noisy, snr)
   - concatenate snr at end: theta = [Ω, SNR]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.utils.torchutils import BoxUniform

from tools.logger_settings import init_logger
from tools.read_params import unpack_vars
from tools.noise import NoiseConfig, apply_noise_policy
from tools.io_compressed import torch_save_compressed

# Priors
from priors.ball_and_sticks import BallAndSticksPriorConfig, build_ball_and_sticks_priors


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_reproducibility(seed: int = 0, num_threads: int = 2) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(num_threads)


def _safe_copy(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Config file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected x with ndim 1 or 2, got shape {tuple(x.shape)}")
    return x


def _make_run_dir(results_root: str | Path, model_name: str) -> Path:
    """
    Create per-run folder:
      results_root/model_name/run_YYYYMMDD_HHMMSS/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / model_name / f"run_{ts}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    return run_dir


# -----------------------------------------------------------------------------
# Prior builder
# -----------------------------------------------------------------------------
def build_prior_from_config(
    forward_model: str,
    prior_args: dict,
    device: str = "cpu",
) -> torch.distributions.Distribution:
    """
    Build Ω prior for the requested forward model.
    Returns a prior over Ω ONLY (no SNR).
    """
    if forward_model == "ball_and_sticks":
        cfg = BallAndSticksPriorConfig(device=device, **prior_args)
        return build_ball_and_sticks_priors(cfg)

    raise ValueError(f"Unknown forward_model: {forward_model}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(config_path: str, *, seed: int = 0, num_threads: int = 2) -> None:
    set_reproducibility(seed=seed, num_threads=num_threads)

    # ---------------------------------------------------------------------
    # Logger created in cwd first, then moved to run_dir/logs
    # ---------------------------------------------------------------------
    start_time = datetime.now()
    init_time = start_time.strftime("%H%M%S")
    log_filename = f"main_log_{init_time}.log"
    logger = init_logger(os.getcwd(), log_filename, __name__)

    logger.info("SBI pipeline started (TRAINING ONLY)")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Seed: {seed} | torch num_threads: {num_threads}")

    # ---------------------------------------------------------------------
    # Read config
    # ---------------------------------------------------------------------
    paths, forward_model_params, training_params, inference_params = unpack_vars(config_path)

    base_path, code_dir, results_root = paths
    forward_model, args_model, prior_type, prior_args, simulator = forward_model_params

    (
        run_training,
        train_data_path,
        density_estimator_type,
        n_train,
        n_jobs_train,
        model_name,
    ) = training_params

    # We are skipping inference for now intentionally
    logger.info("Inference step is skipped in this version.")

    # ---------------------------------------------------------------------
    # Create per-run folder
    # ---------------------------------------------------------------------
    run_dir = _make_run_dir(results_root, model_name)
    logger.info(f"Run directory: {run_dir}")

    # Move log file into run_dir/logs
    log_src = Path(os.getcwd()) / log_filename
    if log_src.exists():
        shutil.move(str(log_src), str(run_dir / "logs" / log_filename))

    # Copy config into run folder
    _safe_copy(config_path, run_dir / "configfile.py")

    # ---------------------------------------------------------------------
    # Noise config
    # Keep noise config in a single place: prior_args["noise_cfg"]
    # (and we pop it out so it doesn't pollute the prior config)
    # ---------------------------------------------------------------------
    noise_cfg_dict = None
    if isinstance(prior_args, dict) and "noise_cfg" in prior_args:
        noise_cfg_dict = prior_args.pop("noise_cfg")

    device = "cpu"

    # ---------------------------------------------------------------------
    # Build Ω priors (no SNR here)
    # ---------------------------------------------------------------------
    priors_omega = build_prior_from_config(
        forward_model=forward_model,
        prior_args=prior_args if isinstance(prior_args, dict) else {},
        device=device,
    )

    noise_cfg: Optional[NoiseConfig] = None
    if noise_cfg_dict is not None:
        noise_cfg = NoiseConfig(device=device, **noise_cfg_dict)
        logger.info(f"Noise policy enabled: type={noise_cfg.noise_type}, strategy={noise_cfg.strategy}")
    else:
        logger.info("Noise policy DISABLED (training will be noisefree; theta=Ω only).")

    # ---------------------------------------------------------------------
    # Dataset saving policy
    # Big-chunk style (your preference), but compressed + float16 where possible.
    # ---------------------------------------------------------------------
    save_x_noisefree: bool = True  # default True (you asked)
    dataset_codec = "lzma"         # "lzma" best compression; "gzip" faster; "none" no compression
    dataset_preset = 6            # 0..9 for lzma

    # =====================================================================
    # TRAINING
    # =====================================================================
    if not run_training:
        logger.info("run_training=False. Nothing to do.")
        return

    logger.info("=== TRAINING PHASE ===")
    logger.info(f"n_train: {n_train} | n_jobs_train: {n_jobs_train}")
    logger.info(f"train_data_path: {train_data_path}")

    simulator_sbi, priors_omega_sbi = prepare_for_sbi(simulator, priors_omega)

    # ------------------------------
    # Load or simulate training data
    # ------------------------------
    if train_data_path is not None:
        # If you load, we assume you already saved:
        # - theta (maybe includes snr)
        # - x (noisy or noisefree depending on your dataset)
        raise NotImplementedError(
            "train_data_path loading is not wired in this training-only refactor yet. "
            "For now, set train_data_path=None to simulate."
        )

    # ------------------------------
    # Simulate noisefree (Ω, x0)
    # ------------------------------
    logger.info("Simulating noisefree training data via sbi.simulate_for_sbi")
    theta_omega, x_noisefree = simulate_for_sbi(
        simulator=simulator_sbi,
        proposal=priors_omega_sbi,
        num_simulations=n_train,
        num_workers=n_jobs_train,
        show_progress_bar=True,
    )
    x_noisefree = _ensure_2d(x_noisefree)

    # ------------------------------
    # Apply noise policy (optional)
    # ------------------------------
    if noise_cfg is not None:
        logger.info("Applying noise policy outside the forward model")

        x_train, snr = apply_noise_policy(x_noisefree, noise_cfg, base_repeat=True)

        # Repeat Ω to match multilevel expansion
        if noise_cfg.strategy == "multilevel":
            theta_omega = theta_omega.repeat(noise_cfg.n_levels, 1)

        # theta = [Ω, SNR] (SNR last, paper convention)
        theta_train = torch.cat([theta_omega, snr.reshape(-1, 1)], dim=1)

        # Shuffle (important for multilevel)
        idx = torch.randperm(x_train.shape[0])
        x_train = x_train[idx]
        theta_train = theta_train[idx]

        # Expand noisefree too if you plan to store it aligned with x_train
        if save_x_noisefree and noise_cfg.strategy == "multilevel":
            x_noisefree_to_save = x_noisefree.repeat(noise_cfg.n_levels, 1)[idx]
        else:
            x_noisefree_to_save = x_noisefree[idx] if save_x_noisefree else None

    else:
        x_train = x_noisefree
        theta_train = theta_omega
        x_noisefree_to_save = x_noisefree if save_x_noisefree else None

    # ------------------------------
    # Save dataset (Torch + float16 + compression)
    # ------------------------------
    dataset_path = run_dir / "data" / "train_dataset.pt.xz"

    payload = {
        "theta": theta_train.to(torch.float32).cpu(),     # keep theta stable
        "x": x_train.to(torch.float16).cpu(),             # big tensor -> float16 on disk
        "meta": {
            "forward_model": forward_model,
            "model_name": model_name,
            "n_train": int(theta_train.shape[0]),
            "signal_dim": int(x_train.shape[1]),
            "theta_dim": int(theta_train.shape[1]),
            "noise_enabled": noise_cfg is not None,
            "noise_cfg": noise_cfg_dict,
            "dtype_on_disk": {"theta": "float32", "x": "float16", "x0": "float16" if save_x_noisefree else None},
            "codec": dataset_codec,
            "preset": dataset_preset,
        },
    }
    if save_x_noisefree and x_noisefree_to_save is not None:
        payload["x0"] = x_noisefree_to_save.to(torch.float16).cpu()

    logger.info(f"Saving dataset to: {dataset_path} (codec={dataset_codec}, preset={dataset_preset})")
    torch_save_compressed(payload, dataset_path, codec=dataset_codec, preset=dataset_preset)

    # ------------------------------
    # Train SNPE-C
    # ------------------------------
    logger.info("Training SNPE-C density estimator")

    # NOTE about prior in SNPE_C:
    # We run single-round amortised training (no sequential proposal updates),
    # so the learned mapping is driven by (theta_train, x_train).
    #
    # Still, sbi requires a prior object, and it must match theta dimension.
    # Easiest robust choice: BoxUniform over the observed theta range.
    theta_min = theta_train.min(dim=0).values
    theta_max = theta_train.max(dim=0).values
    prior_for_snpe = BoxUniform(low=theta_min, high=theta_max)

    trainer = SNPE_C(
        prior=prior_for_snpe,
        density_estimator=density_estimator_type,
        device=device,
        logging_level="INFO",
        show_progress_bars=True,
    )

    # Cast x back to float32 for training stability
    x_train_f32 = x_train.float()
    theta_train_f32 = theta_train.float()

    density_estimator = (
        trainer.append_simulations(theta_train_f32, x_train_f32)
        .train(use_combined_loss=True, show_train_summary=True)
    )
    posterior = trainer.build_posterior()

    # Save model artifacts inside run folder
    model_dir = run_dir / "models"
    torch.save(density_estimator, model_dir / "density_estimator.pt")
    torch.save(posterior, model_dir / "posterior.pt")

    logger.info("Training completed")
    logger.info(f"Saved: {model_dir / 'density_estimator.pt'}")
    logger.info(f"Saved: {model_dir / 'posterior.pt'}")
    logger.info("SBI pipeline finished successfully")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SBI training (training-only) from a Python config file.")
    parser.add_argument("configfile", type=str, help="Path to configfile.py")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy/torch/random.")
    parser.add_argument("--threads", type=int, default=2, help="Number of torch CPU threads.")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    main(args.configfile, seed=args.seed, num_threads=args.threads)