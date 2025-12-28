from pathlib import Path


def _save_example_signals(
    y,
    title: str,
    filename: str,
    max_curves: int = 6,
):
    """
    Save a quick visual sanity-check plot of attenuation signals.

    Parameters
    ----------
    y : torch.Tensor
        Shape (N, G)
    title : str
        Figure title.
    filename : str
        Output filename (saved under test_plots/).
    max_curves : int
        Number of example signals to plot.
    """
    import matplotlib.pyplot as plt
    import torch

    outdir = Path("test_plots")
    outdir.mkdir(exist_ok=True)

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    n = min(max_curves, y.shape[0])

    plt.figure(figsize=(6, 4))
    for i in range(n):
        plt.plot(y[i], alpha=0.8)

    plt.xlabel("Gradient index")
    plt.ylabel("Attenuation S / S0")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outfile = outdir / filename
    plt.savefig(outfile, dpi=150)
    plt.close()

    print(f"[saved] {outfile}")