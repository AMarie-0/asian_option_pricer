from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_window_sensitivity(
    df: pd.DataFrame,
    ticker: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side bar charts: option price and sigma for each volatility window."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    windows = df["window"].tolist()
    colors  = ["#4a9eff", "#f0a500", "#50c878", "#e05050"]

    axes[0].barh(windows, df["price"], color=colors, alpha=0.85)
    axes[0].set_title("Option Price by Window", color="white", fontsize=11)
    axes[0].set_xlabel("Price ($)", color="white")
    axes[0].tick_params(colors="white")
    axes[0].spines[:].set_color("#333")
    for bar, val in zip(axes[0].patches, df["price"]):
        axes[0].text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                     f"${val:.3f}", va="center", color="white", fontsize=9)

    axes[1].barh(windows, df["sigma_pct"], color=colors, alpha=0.85)
    axes[1].set_title("Volatility σ by Window", color="white", fontsize=11)
    axes[1].set_xlabel("σ (%)", color="white")
    axes[1].tick_params(colors="white")
    axes[1].spines[:].set_color("#333")
    for bar, val in zip(axes[1].patches, df["sigma_pct"]):
        axes[1].text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}%", va="center", color="white", fontsize=9)

    fig.suptitle(f"{ticker} — Volatility Window Sensitivity", color="white", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_convergence(
    df: pd.DataFrame,
    ticker: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of option price vs number of binomial steps."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.plot(df["n"], df["price"], marker="o", color="#4a9eff",
            lw=2, ms=5, label="Binomial Price")
    ax.axhline(df["price"].iloc[-1], color="#f0a500", ls="--", lw=1,
               label=f"Limit ≈ ${df['price'].iloc[-1]:.4f}")

    ax.set_title(f"{ticker} — Price Convergence vs Steps", color="white", fontsize=13)
    ax.set_xlabel("Number of Steps (n)", color="white")
    ax.set_ylabel("Option Price ($)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
