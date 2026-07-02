from __future__ import annotations

from copy import copy
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.model.calibration import calibrate
from src.model.pricer import price_asian_call


def plot_price_surface(
    df: pd.DataFrame,
    ticker: str,
    sigma_range: tuple = (0.10, 0.60),
    r_range:     tuple = (0.0001, 0.0005),
    grid_size:   int   = 30,
    n: int = 25,
) -> go.Figure:
    """3D surface of option price as a function of (sigma, r)."""
    sigmas = np.linspace(*sigma_range, grid_size)
    rates  = np.linspace(*r_range,     grid_size)
    Z      = np.zeros((grid_size, grid_size))

    base = calibrate(df, ticker, n=n)

    for i, sigma in enumerate(sigmas):
        for j, r in enumerate(rates):
            p       = copy(base)
            p.sigma = sigma
            p.r     = r
            dt      = p.T / p.n
            p.u     = np.exp(sigma * np.sqrt(dt))
            p.d     = 1.0 / p.u
            r_per   = (1 + r) ** (252 * dt) - 1
            p.q     = (1 + r_per - p.d) / (p.u - p.d)
            Z[i, j] = price_asian_call(p)["price"]

    fig = go.Figure(data=[go.Surface(
        x=rates * 100,
        y=sigmas * 100,
        z=Z,
        colorscale="Viridis",
        colorbar=dict(title="Option Price ($)"),
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
