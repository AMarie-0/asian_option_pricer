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
    ticker:   str
    S0:       float
    sigma:    float   # annualised volatility
    u:        float
    d:        float
    q:        float   # risk-neutral probability
    r:        float   # annual risk-free rate
    n:        int
    dt:       float   # years per step  (T/n)
    T:        float   # maturity in years
    R_period: float   # 1 + r*dt  (one-step gross return, simple discrete)
    discount: float   # 1 / R_period

    def summary(self) -> str:
        lines = [
            "\n-- Model Parameters ------------------------------------------",
            f"  Ticker         = {self.ticker}",
            f"  S0             = {self.S0:.4f}  USD",
            f"  sigma (annual) = {self.sigma:.6f}  ({self.sigma*100:.4f}%)",
            f"  r     (annual) = {self.r:.6f}  ({self.r*100:.4f}%)",
            f"  T              = {self.T:.4f}  years",
            f"  n              = {self.n}  steps",
            f"  dt             = {self.dt:.6f}  years/step",
            f"  u              = {self.u:.6f}",
            f"  d              = {self.d:.6f}",
            f"  R_period       = {self.R_period:.8f}",
            f"  q  (RN prob)   = {self.q:.6f}",
            f"  discount       = {self.discount:.8f}",
            "--------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def display(self) -> None:
        print(self.summary())


def calibrate(
    df: pd.DataFrame,
    ticker: str = "",
    r: float = 0.01,          # annual rate (e.g. 0.01 = 1%)
    n: int = 25,
    T: float = 0.5,
    price_col: str = "adj_close",
    volatility_window: str = "full",
) -> ModelParams:
    """
    Estimate CRR parameters from historical adjusted closing prices.

    Convention (matches R code exactly):
      - sigma annualised with sqrt(250) trading days
      - R_period = 1 + r*dt  (simple discrete per-step)
      - q = (R_period - d) / (u - d)
      - discount = 1 / R_period
    """
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

    log_returns  = np.diff(np.log(prices))
    sigma_daily  = float(np.std(log_returns, ddof=1))
    sigma_annual = sigma_daily * math.sqrt(250)   # 250 trading days, matches R

    dt       = T / n
    u        = math.exp(sigma_annual * math.sqrt(dt))
    d        = 1.0 / u

    R_period = 1.0 + r * dt        # simple discrete, matches R code
    q        = (R_period - d) / (u - d)
    discount = 1.0 / R_period

    if not (0.0 < q < 1.0):
        raise ValueError(
            f"Risk-neutral probability q={q:.6f} outside (0,1). "
            "Check r, sigma, and dt satisfy no-arbitrage: d < R_period < u."
        )

    S0 = float(prices[-1])

    return ModelParams(
        ticker=ticker, S0=S0, sigma=sigma_annual,
        u=u, d=d, q=q, r=r, n=n, dt=dt, T=T,
        R_period=R_period, discount=discount,
    )
