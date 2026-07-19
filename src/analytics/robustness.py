from __future__ import annotations

import math
import dataclasses
import numpy as np
import pandas as pd

from src.model.calibration import calibrate, ModelParams
from src.model.pricer import price_asian_call
from src.analytics.approximation import normal_approximation_price

WINDOWS = {
    "full":       ("2020–2026 (Baseline)",    None),
    "post_covid": ("2021–2026 (Post-COVID)",  "2021-01-01"),
    "3y":         ("2023–2026 (3-Year)",      "2023-01-01"),
    "1y":         ("2025–2026 (1-Year)",      "2025-01-01"),
}


def volatility_window_sensitivity(
    df: pd.DataFrame,
    ticker: str,
    r: float = 0.01,
    T: float = 0.5,
    n: int   = 25,
) -> pd.DataFrame:
    """
    Re-price across four historical volatility estimation windows.
    All other parameters (r, T, n, S0) are held fixed at baseline values.
    Mirrors Section 6 of the report.
    """
    rows = []
    baseline_price = None

    for key, (label, cutoff) in WINDOWS.items():
        sub = df.copy()
        if cutoff is not None and "date" in sub.columns:
            sub = sub[pd.to_datetime(sub["date"]) >= pd.Timestamp(cutoff)]

        prices     = sub["adj_close"].dropna().values
        log_ret    = np.diff(np.log(prices))
        sig_daily  = float(np.std(log_ret, ddof=1))
        sig_annual = sig_daily * math.sqrt(250)
        n_obs      = len(log_ret)

        params = calibrate(sub, ticker, r=r, T=T, n=n, volatility_window=key)
        result = price_asian_call(params)

        row = {
            "window":           label,
            "n_obs":            n_obs,
            "sigma_daily_pct":  sig_daily  * 100,
            "sigma_annual_pct": sig_annual * 100,
            "price":            result["price"],
        }
        rows.append(row)
        if key == "full":
            baseline_price = result["price"]

    out = pd.DataFrame(rows)
    out["delta_vs_baseline"] = out["price"] - baseline_price
    out["delta_pct"]         = (out["price"] - baseline_price) / baseline_price * 100
    return out


def normal_approx_convergence(
    p: ModelParams,
    n_grid: list[int] | None = None,
) -> pd.DataFrame:
    """
    V0_approx as a function of n (Figure 10 in the report).
    Holds S0, sigma, r, T fixed and varies only n.
    Shows convergence toward the asymptote e^{-rT}*S0*sigma*sqrt(T/3)/sqrt(2*pi).
    """
    if n_grid is None:
        n_grid = list(range(5, 101, 5))

    rows = []
    for n_val in n_grid:
        dt       = p.T / n_val
        u        = math.exp(p.sigma * math.sqrt(dt))
        d        = 1.0 / u
        R_period = 1.0 + p.r * dt
        q        = (R_period - d) / (u - d)
        p_n = dataclasses.replace(
            p,
            n=n_val, dt=dt, u=u, d=d, q=q,
            R_period=R_period, discount=1.0 / R_period,
        )
        approx = normal_approximation_price(p_n)
        rows.append({
            "n":            n_val,
            "price_approx": approx["price"],
            "price_limit":  approx["price_limit"],
        })

    return pd.DataFrame(rows)
