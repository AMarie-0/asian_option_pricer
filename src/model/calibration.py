from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_CUTOFFS: dict[str, str | None] = {
    "full":       None,
    "post_covid": "2021-01-01",
    "3y":         "2023-01-01",
    "1y":         "2025-01-01",
}


@dataclass
class ModelParams:
    ticker: str
    S0:     float
    sigma:  float
    u:      float
    d:      float
    q:      float
    r:      float
    n:      int
    dt:     float
    T:      float

    def summary(self) -> str:
        r_daily = self.r / 250
        lines = [
            "\n-- Model Parameters -----------------------------------------",
            f"  Ticker         = {self.ticker}",
            f"  S0             = {self.S0:.4f}  USD",
            f"  sigma (annual) = {self.sigma:.6f}  ({self.sigma*100:.4f}% p.a.)",
            f"  r (annual)     = {self.r:.6f}  ({self.r*100:.4f}% p.a.)",
            f"  r (daily)      = {r_daily:.8f}  ({r_daily*100:.6f}% per day)",
            f"  T              = {self.T:.4f}  years",
            f"  n              = {self.n}  steps",
            f"  dt             = {self.dt:.6f}  years/step",
            f"  u              = {self.u:.6f}",
            f"  d              = {self.d:.6f}",
            f"  q  (RN prob)   = {self.q:.6f}",
            "-----------------------------------------------------",
        ]
        return "\n".join(lines)

    def display(self) -> None:
        print(self.summary())


def calibrate(
    df: pd.DataFrame,
    ticker: str = "",
    r: float = 0.01,        # annual continuous risk-free rate
    n: int = 25,
    T: float = 0.5,
    price_col: str = "adj_close",
    volatility_window: str = "full",
) -> ModelParams:
    cutoff = WINDOW_CUTOFFS.get(volatility_window)
    if cutoff is not None and "date" in df.columns:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(cutoff)]

    if price_col not in df.columns:
        raise ValueError(
            f"Column '{price_col}' not found. Available: {list(df.columns)}"
        )

    prices = df[price_col].dropna().values
    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations.")

    log_returns = np.diff(np.log(prices))
    n_obs       = len(log_returns)

    sigma_daily  = float(np.std(log_returns, ddof=1))
    sigma_annual = sigma_daily * math.sqrt(250)   # 250 trading days per year (assignment spec)

    dt = T / n
    u  = math.exp(sigma_annual * math.sqrt(dt))
    d  = 1.0 / u

    e_r_dt = math.exp(r * dt)
    q      = (e_r_dt - d) / (u - d)

    if not (0 < q < 1):
        raise ValueError(
            f"Risk-neutral probability q={q:.6f} outside (0,1). "
            "Check r, σ, and Δt satisfy no-arbitrage condition."
        )

    S0 = float(prices[-1])

    print(f"[calibrate] {ticker or 'unknown'} | {n_obs} log-returns | "
          f"sigma_daily={sigma_daily:.6f} | sigma_annual={sigma_annual:.6f}")

    return ModelParams(
        ticker=ticker,
        S0=S0, sigma=sigma_annual,
        u=u, d=d, q=q,
        r=r, n=n, dt=dt, T=T,
    )
