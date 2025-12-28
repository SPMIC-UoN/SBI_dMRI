"""
Factory functions for constructing Ball-and-Sticks simulators.

This module is responsible ONLY for:
- reading acquisition information from a config dictionary
- instantiating the correct simulator objects

It deliberately contains no training or inference logic.
"""

from models.ball_and_sticks.simulator import (
    GradientTable,
    BallAndSticksAttenuation,
)


def build_ball_and_sticks_simulator(cfg: dict, device: str = "cpu") -> BallAndSticksAttenuation:
    """
    Build a Ball-and-Sticks attenuation simulator from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Dictionary obtained from executing a configfile.py.
        Must contain:
            - 'bvals_path'
            - 'bvecs_path'
    device : str
        Torch device (e.g. 'cpu', 'cuda').

    Returns
    -------
    BallAndSticksAttenuation
        Callable simulator compatible with sbi.simulate_for_sbi.
    """

    # Read diffusion gradient table (attenuation model: no b0 volumes)
    gtab = GradientTable.read_bvals_bvecs(
        bvals_path=cfg["bvals_path"],
        bvecs_path=cfg["bvecs_path"],
        device=device,
    )

    # Construct the simulator
    simulator = BallAndSticksAttenuation(
        gtab=gtab,
        device=device,
    )

    return simulator