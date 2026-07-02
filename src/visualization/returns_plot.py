from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_log_returns(
    df: pd.DataFrame,
    ticker: str,
    sigma: float,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot daily log returns with ±1σ bands."""
    prices  = df["adj_close"].values
    log_ret = np.diff(np.log(prices))
    dates   = pd.to_datetime(df["date"].iloc[1:]).values

    sigma_daily = sigma / np.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.plot(dates, log_ret, color="#4a9eff", lw=0.7, alpha=0.85, label="Log Returns")
    ax.axhline( sigma_daily, color="#f0a500", ls="--", lw=1.2, label="+1σ daily")
    ax.axhline(-sigma_daily, color="#e05050", ls="--", lw=1.2, label="-1σ daily")
    ax.axhline(0, color="white", lw=0.4, alpha=0.4)

    ax.fill_between(dates, -sigma_daily, sigma_daily, alpha=0.07, color="white")

    ax.set_title(f"{ticker} — Daily Log Returns", color="white", fontsize=13)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Log Return", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
