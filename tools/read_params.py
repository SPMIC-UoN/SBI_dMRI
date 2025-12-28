"""
tools/read_params.py

Load a Python config file and return:
  paths, forward_model_params, training_params, inference_params

Config must define these names (dict-style recommended):

paths = dict(base_path=..., code_dir=..., results_root=...)
forward_model_params = dict(
    forward_model="ball_and_sticks",
    args_model=dict(...),
    prior_type="nonrestricted",
    prior_args=dict(...),
    simulator=<callable>,
)

training_params = dict(
    run_training=True,
    train_data_path=None,
    density_estimator_type="nsf",
    n_train=1_000_000,
    n_jobs_train=8,
    model_name="...",
)

inference_params = dict(...)   # can be present but unused for now
"""

from __future__ import annotations

from pathlib import Path
from runpy import run_path
from typing import Any, Dict, Tuple


def unpack_vars(config_path: str) -> Tuple[tuple, tuple, tuple, tuple]:
    cfg_path = Path(config_path)
    if not cfg_path.exists() or cfg_path.suffix != ".py":
        raise FileNotFoundError(f"Config file not found or not .py: {cfg_path}")

    args: Dict[str, Any] = run_path(str(cfg_path))

    # ---- paths ----
    if "paths" not in args:
        raise KeyError("Config must define `paths` dict.")
    paths_d = args["paths"]
    base_path = paths_d["base_path"]
    code_dir = paths_d["code_dir"]
    results_root = paths_d["results_root"]
    paths = (base_path, code_dir, results_root)

    # ---- forward_model_params ----
    if "forward_model_params" not in args:
        raise KeyError("Config must define `forward_model_params` dict.")
    fwd = args["forward_model_params"]
    required_fwd = ["forward_model", "args_model", "prior_type", "prior_args", "simulator"]
    for k in required_fwd:
        if k not in fwd:
            raise KeyError(f"forward_model_params missing key: {k}")

    forward_model_params = (
        fwd["forward_model"],
        fwd["args_model"],
        fwd["prior_type"],
        fwd["prior_args"],
        fwd["simulator"],
    )

    # ---- training_params ----
    if "training_params" not in args:
        raise KeyError("Config must define `training_params` dict.")
    tr = args["training_params"]
    required_tr = ["run_training", "train_data_path", "density_estimator_type", "n_train", "n_jobs_train", "model_name"]
    for k in required_tr:
        if k not in tr:
            raise KeyError(f"training_params missing key: {k}")

    training_params = (
        tr["run_training"],
        tr["train_data_path"],
        tr["density_estimator_type"],
        int(tr["n_train"]),
        int(tr["n_jobs_train"]),
        tr["model_name"],
    )

    # ---- inference_params (optional but returned for API compatibility) ----
    inf_d = args.get("inference_params", dict(
        run_inference=False,
        pretrained_model_path=None,
        obs_file=None,
        mask_file=None,
        n_posterior_samples=500,
        n_jobs_inference=1,
    ))
    inference_params = (
        bool(inf_d.get("run_inference", False)),
        inf_d.get("pretrained_model_path", None),
        inf_d.get("obs_file", None),
        inf_d.get("mask_file", None),
        int(inf_d.get("n_posterior_samples", 500)),
        int(inf_d.get("n_jobs_inference", 1)),
    )

    return paths, forward_model_params, training_params, inference_params