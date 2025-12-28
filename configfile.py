device = "cpu"

bvals_path = "/path/to/model2/bvals_nob0s"
bvecs_path = "/path/to/model2/bvecs_nob0s"

nfib = 3
modelnum = 2

noise_cfg = dict(
    noise_type="rician",        # gaussian | rician
    strategy="multilevel",      # random | multilevel | fixed  (used only if you *sample* SNR)
    snr_min=3.0,
    snr_max=80.0,
    n_levels=8,
    snr_fixed=30.0,
    snr_fixed_jitter=0.0,
    device=device,
)

prior_args = dict(
    nfib=nfib,
    modelnum=modelnum,
    hemisphere=True,
    include_snr=True,     # <--- key: priors should sample SNR too
    snr_min=noise_cfg["snr_min"],
    snr_max=noise_cfg["snr_max"],
)

args_model = dict(
    bvals_path=bvals_path,
    bvecs_path=bvecs_path,
    nfib=nfib,
    modelnum=modelnum,
    device=device,
    include_snr_in_theta=True,   # <--- key: theta = [phys..., SNR]
    noise_cfg=noise_cfg,         # to set noise_type for corruption
)

forward_model_params = (
    "ball_and_sticks",
    args_model,
    "ball_and_sticks",     # prior_type
    prior_args,
    "factory",             # your pipeline will interpret this and call make_ball_and_sticks_simulator
)