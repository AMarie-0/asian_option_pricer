# data/calibrate.py
"""
Calibrate binomial-tree parameters from a price series.

Exports
-------
ModelParams   – dataclass holding all model inputs
calibrate     – function that produces a ModelParams from a price DataFrame
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── 1. Parameter container ────────────────────────────────────────────────────

@dataclass
class ModelParams:
    S0:    float   # last adjusted closing price
    sigma: float   # annualised volatility  (σ)
    u:     float   # up factor              e^(σ√Δt)
    d:     float   # down factor            1/u
    q:     float   # risk-neutral prob      (e^(rΔt) - d) / (u - d)
    r:     float   # continuously-compounded risk-free rate (annual)
    n:     int     # number of time steps
    dt:    float   # length of one step in years  (T/n)
    T:     float   # option maturity in years

    def display(self) -> None:
        print("\n── Model Parameters ─────────────────────────────────")
        print(f"  S0            = {self.S0:.4f}  USD")
        print(f"  σ  (annual)   = {self.sigma:.6f}  ({self.sigma*100:.4f}%)")
        print(f"  r  (annual)   = {self.r:.6f}  ({self.r*100:.4f}%)")
        print(f"  T             = {self.T:.4f}  years")
        print(f"  n             = {self.n}  steps")
        print(f"  Δt            = {self.dt:.6f}  years/step")
        print(f"  u             = {self.u:.6f}")
        print(f"  d             = {self.d:.6f}")
        print(f"  q  (RN prob)  = {self.q:.6f}")
        print("─────────────────────────────────────────────────────\n")


# ── 2. Main calibration function ──────────────────────────────────────────────

def calibrate(
    df: pd.DataFrame,
    r: float,
    n: int   = 25,
    T: float = 25 / 252,        # 25 trading days ≈ one month
    price_col: str = "adj_close",
) -> ModelParams:
    """
    Parameters
    ----------
    df        : DataFrame with at least a `price_col` column, sorted by date.
    r         : annualised continuously-compounded risk-free rate.
    n         : number of binomial steps (default 25).
    T         : option maturity in years (default 25/252 ≈ one trading month).
    price_col : column name for adjusted closing prices.

    Returns
    -------
    ModelParams
    """

    # ── a. Validate input ────────────────────────────────────────────────────
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame. "
                         f"Available: {list(df.columns)}")

    prices = df[price_col].dropna().values
    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations to compute returns.")

    # ── b. Log returns ───────────────────────────────────────────────────────
    log_returns = np.diff(np.log(prices))           # r_t = log(P_t / P_{t-1})
    n_obs       = len(log_returns)

    # ── c. Volatility ────────────────────────────────────────────────────────
    sigma_daily  = float(np.std(log_returns, ddof=1))   # sample std
    sigma_annual = sigma_daily * math.sqrt(252)         # annualise

    # ── d. Binomial factors ───────────────────────────────────────────────────
    dt = T / n
    u  = math.exp(sigma_annual * math.sqrt(dt))
    d  = 1.0 / u                                        # recombining tree

    # ── e. Risk-neutral probability ───────────────────────────────────────────
    # q = (e^(r·Δt) - d) / (u - d)
    e_r_dt = math.exp(r * dt)
    q      = (e_r_dt - d) / (u - d)

    if not (0 < q < 1):
        raise ValueError(
            f"Risk-neutral probability q={q:.6f} is outside (0,1). "
            "Check that r, σ, and Δt satisfy the no-arbitrage condition."
        )

    # ── f. Current price ──────────────────────────────────────────────────────
    S0 = float(prices[-1])   # last observation = pricing date

    print(f"[calibrate] {n_obs} log-returns used.")
    print(f"[calibrate] σ_daily = {sigma_daily:.6f} | σ_annual = {sigma_annual:.6f}")

    return ModelParams(
        S0    = S0,
        sigma = sigma_annual,
        u     = u,
        d     = d,
        q     = q,
        r     = r,
        n     = n,
        dt    = dt,
        T     = T,
    )
