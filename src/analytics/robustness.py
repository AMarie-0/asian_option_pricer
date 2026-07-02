from __future__ import annotations

import pandas as pd
from src.model.calibration import calibrate
from src.model.pricer import price_asian_call

WINDOWS = ["full", "post_covid", "3y", "1y"]
WINDOW_LABELS = {
    "full":       "2020–2026 (Baseline)",
    "post_covid": "2021–2026 (Post-COVID)",
    "3y":         "2023–2026 (3-Year)",
    "1y":         "2025–2026 (1-Year)",
}


def volatility_window_sensitivity(
    df: pd.DataFrame,
    ticker: str,
    r: float = 0.01,
    T: float = 0.5,
    n: int   = 25,
) -> pd.DataFrame:
    """Re-price the option across four estimation windows."""
    results = []
    baseline_price = None

    for window in WINDOWS:
        params = calibrate(df, ticker, r=r, T=T, n=n, volatility_window=window)
        result = price_asian_call(params)
        row = {
            "window":    WINDOW_LABELS[window],
            "sigma_pct": params.sigma * 100,
            "price":     result["price"],
        }
        results.append(row)
        if window == "full":
            baseline_price = result["price"]

    out = pd.DataFrame(results)
    out["delta_pct"] = (out["price"] - baseline_price) / baseline_price * 100
    return out


def step_convergence(
    df: pd.DataFrame,
    ticker: str,
    r: float = 0.01,
    T: float = 0.5,
    step_grid: list[int] | None = None,
) -> pd.DataFrame:
    """Price vs number of time steps — convergence analysis."""
    if step_grid is None:
        step_grid = [5, 10, 15, 20, 25, 30]   # n>30 can be very slow; normal approx covers convergence beyond

    rows = []
    for n in step_grid:
        params = calibrate(df, ticker, r=r, T=T, n=n)
        result = price_asian_call(params)
        rows.append({"n": n, "price": result["price"], "n_states": result["n_states"]})
    return pd.DataFrame(rows)
