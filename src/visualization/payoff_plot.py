from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from src.model.calibration import ModelParams


N_BINS = 300


def plot_payoff_distribution(
    result: dict,
    params: ModelParams,
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of payoff distribution weighted by risk-neutral probability."""
    payoffs = np.asarray(result["payoffs"])
    weights = np.asarray(result["weights"])
    price   = result["price"]

    hist_w, edges = np.histogram(payoffs, bins=N_BINS, weights=weights)
    centers       = 0.5 * (edges[:-1] + edges[1:])
    bar_width     = (edges[1] - edges[0]) * 0.9
    colors        = np.where(centers > 0, "#4a9eff", "#555")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.bar(centers, hist_w, color=colors, width=bar_width)
    ax.axvline(
        x=price,
        color="orange", lw=1.5, ls="--",
        label=f"Price = ${price:.3f}",
    )

    ax.set_title(
        f"{params.ticker} — Terminal Payoff Distribution "
        f"({result['n_states']:,} states, {N_BINS} bins)",
        color="white", fontsize=12,
    )
    ax.set_xlabel("Payoff ($)", color="white")
    ax.set_ylabel("Risk-Neutral Probability", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
