from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from src.model.calibration import ModelParams
from src.model.binomial_tree import build_stock_tree


def plot_binomial_tree(
    p: ModelParams,
    max_steps: int = 10,
    figsize: tuple = (14, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Draw the recombining stock price tree up to max_steps."""
    steps = min(p.n, max_steps)
    tree  = build_stock_tree(replace(p, n=steps))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    prices_all = [tree[i, j] for i in range(steps + 1) for j in range(i + 1)]
    vmin, vmax = min(prices_all), max(prices_all)
    cmap = plt.get_cmap("RdYlGn")

    for i in range(steps):
        for j in range(i + 1):
            x0, y0 = i, j - i / 2
            ax.plot([x0, i + 1], [y0, (j + 1) - (i + 1) / 2], color="#444", lw=0.8, zorder=1)
            ax.plot([x0, i + 1], [y0, j       - (i + 1) / 2], color="#444", lw=0.8, zorder=1)

    for i in range(steps + 1):
        for j in range(i + 1):
            price = tree[i, j]
            y     = j - i / 2
            color = cmap((price - vmin) / (vmax - vmin))
            ax.scatter(i, y, color=color, s=120, zorder=3)
            ax.text(i, y + 0.18, f"${price:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="white")

    ax.set_title(
        f"{p.ticker} — Binomial Tree (first {steps} of {p.n} steps)",
        color="white", fontsize=13,
    )
    ax.set_xlabel("Time Step", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
