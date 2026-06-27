# src/visualization/tree_plot.py
import numpy as np
import matplotlib.pyplot as plt
from src.model.calibration import ModelParams
from src.model.binomial_tree import build_stock_tree

def plot_binomial_tree(
    p: ModelParams,
    max_steps: int = 10,
    figsize: tuple = (14, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Draw the recombining stock price tree up to max_steps.
    Nodes are coloured by price level (green=high, red=low).
    """
    steps = min(p.n, max_steps)
    tree  = build_stock_tree(ModelParams(
        **{**p.__dict__, "n": steps}
    ))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    prices_all = [tree[i, j] for i in range(steps+1)
                              for j in range(i+1)]
    vmin, vmax = min(prices_all), max(prices_all)
    cmap = plt.get_cmap("RdYlGn")

    # Draw edges first
    for i in range(steps):
        for j in range(i + 1):
            x0, y0 = i,  j - i / 2
            ax.plot([x0, i+1], [y0, (j+1) - (i+1)/2],
                    color="#444", lw=0.8, zorder=1)
            ax.plot([x0, i+1], [y0, j   - (i+1)/2],
                    color="#444", lw=0.8, zorder=1)

    # Draw nodes
    for i in range(steps + 1):
        for j in range(i + 1):
            price = tree[i, j]
            y     = j - i / 2
            color = cmap((price - vmin) / (vmax - vmin))
            ax.scatter(i, y, color=color, s=120, zorder=3)
            ax.text(i, y + 0.18, f"${price:.1f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="white")

    ax.set_title(
        f"{p.ticker} — Binomial Tree (first {steps} of {p.n} steps)",
        color="white", fontsize=13
    )
    ax.set_xlabel("Time Step", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

# src/visualization/surface_plot.py
import numpy as np
import plotly.graph_objects as go
from src.model.calibration import calibrate, ModelParams
from src.model.pricer import price_asian_call

def plot_price_surface(
    df,
    ticker: str,
    sigma_range: tuple = (0.10, 0.60),
    r_range:     tuple = (0.0001, 0.0005),
    grid_size:   int   = 30,
    n: int = 25,
) -> go.Figure:
    """
    3D surface of option price as a function of (sigma, r).
    """
    sigmas = np.linspace(*sigma_range, grid_size)
    rates  = np.linspace(*r_range,     grid_size)
    Z      = np.zeros((grid_size, grid_size))

    base = calibrate(df, ticker, n=n)

    for i, sigma in enumerate(sigmas):
        for j, r in enumerate(rates):
            from copy import copy
            p = copy(base)
            p.sigma = sigma
            p.r     = r
            dt  = p.T / p.n
            p.u = np.exp(sigma * np.sqrt(dt))
            p.d = 1.0 / p.u
            r_period = (1 + r) ** (252 * dt) - 1
            p.q = (1 + r_period - p.d) / (p.u - p.d)
            Z[i, j] = price_asian_call(p)["price"]

    fig = go.Figure(data=[go.Surface(
        x=rates * 100,
        y=sigmas * 100,
        z=Z,
        colorscale="Viridis",
        colorbar=dict(title="Option Price ($)")
    )])
    fig.update_layout(
        title=f"{ticker} — Asian Call Price Surface",
        scene=dict(
            xaxis_title="Risk-Free Rate r (%)",
            yaxis_title="Volatility σ (%)",
            zaxis_title="Option Price ($)",
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font_color="white",
        height=650,
    )
    return fig
