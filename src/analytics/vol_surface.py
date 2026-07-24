"""
vol_surface.py — Implied volatility surface from live options chains.

What this shows
---------------
The pricing model in this tool uses ONE constant sigma estimated from
historical returns. The market does not. Every listed option implies its
own sigma via Black-Scholes inversion, and those implied vols vary
systematically with strike and maturity:

  SKEW / SMILE  — OTM puts trade at higher IV than ATM options. Equity
                  markets price crash risk, so the left tail is fatter
                  than lognormal. This is the single largest empirical
                  failure of Black-Scholes.

  TERM STRUCTURE— IV varies with maturity: typically upward-sloping in
                  calm markets (uncertainty compounds), inverted during
                  stress (near-term panic).

Plotting the surface next to the flat historical sigma makes the size of
the modelling simplification visible.

Implementation
--------------
1. Fetch option chains for several expiries via yfinance.
2. Filter out illiquid junk (no volume, zero bid, absurd spreads) — bad
   quotes produce nonsense IVs and wreck the surface.
3. Invert Black-Scholes for each surviving quote using Brent's method on
   the bracketed interval [1e-6, 5.0]. Brent is used rather than
   Newton-Raphson because vega collapses to zero for deep ITM/OTM options,
   which makes Newton unstable exactly where quotes are least reliable.

Everything degrades gracefully: if yfinance has no chain for a ticker,
the caller gets an empty frame and the UI shows an explanatory message.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S0: float, K: float, r: float, sigma: float, T: float,
             kind: str = "call") -> float:
    """Black-Scholes price with continuous discounting."""
    if T <= 0 or sigma <= 0:
        intrinsic = (max(S0 - K, 0.0) if kind == "call" else max(K - S0, 0.0))
        return intrinsic
    sqT = sigma * math.sqrt(T)
    d1  = (math.log(S0 / K) + (r + sigma ** 2 / 2) * T) / sqT
    d2  = d1 - sqT
    disc = math.exp(-r * T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def implied_vol(market_price: float, S0: float, K: float, r: float,
                T: float, kind: str = "call") -> float:
    """
    Invert Black-Scholes for sigma using Brent's method.

    Returns NaN when the quote admits no solution — typically because the
    price violates arbitrage bounds (below intrinsic value, above spot),
    which happens with stale or crossed quotes.
    """
    if T <= 0 or market_price <= 0:
        return float("nan")

    # No-arbitrage bounds; outside them no sigma can reproduce the price
    disc = math.exp(-r * T)
    lower = (max(S0 - K * disc, 0.0) if kind == "call"
             else max(K * disc - S0, 0.0))
    upper = S0 if kind == "call" else K * disc
    if not (lower < market_price < upper):
        return float("nan")

    try:
        return brentq(
            lambda s: bs_price(S0, K, r, s, T, kind) - market_price,
            1e-6, 5.0, maxiter=100, xtol=1e-8,
        )
    except (ValueError, RuntimeError):
        return float("nan")


def fetch_vol_surface(
    ticker: str, r: float = 0.04, max_expiries: int = 8,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
    min_volume: int = 1, kind: str = "call",
) -> pd.DataFrame:
    """
    Build the implied-vol surface for `ticker` from live option chains.

    Returns a tidy DataFrame with one row per surviving quote:
      expiry, T (years), strike, moneyness (K/S0), mid price, implied_vol,
      volume, open_interest

    An empty frame means no usable quotes were found — the caller should
    show a message rather than an empty chart.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)

    try:
        expiries = tk.options
    except Exception:
        return pd.DataFrame()
    if not expiries:
        return pd.DataFrame()

    # Spot: prefer live price, fall back to last close
    S0 = None
    try:
        S0 = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
    except Exception:
        pass
    if not S0:
        try:
            S0 = float(tk.history(period="5d")["Close"].iloc[-1])
        except Exception:
            return pd.DataFrame()
    S0 = float(S0)

    today = datetime.now().date()
    rows = []

    for exp in expiries[:max_expiries]:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            T = (exp_date - today).days / 365.0
            if T <= 0.01:          # skip same-week expiries: IV is noise
                continue
            chain = tk.option_chain(exp)
            quotes = chain.calls if kind == "call" else chain.puts
        except Exception:
            continue

        for _, q in quotes.iterrows():
            try:
                K = float(q["strike"])
                m = K / S0
                if not (moneyness_range[0] <= m <= moneyness_range[1]):
                    continue

                bid, ask = float(q.get("bid", 0) or 0), float(q.get("ask", 0) or 0)
                vol = int(q.get("volume", 0) or 0)
                oi  = int(q.get("openInterest", 0) or 0)

                # Liquidity filter — bad quotes give meaningless IVs
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                if vol < min_volume and oi < 10:
                    continue
                if (ask - bid) / ((ask + bid) / 2) > 0.6:   # spread > 60%
                    continue

                mid = (bid + ask) / 2
                iv = implied_vol(mid, S0, K, r, T, kind)
                if not np.isfinite(iv) or iv < 0.01 or iv > 3.0:
                    continue

                rows.append({
                    "expiry": exp, "T": T, "strike": K, "moneyness": m,
                    "mid_price": mid, "implied_vol": iv,
                    "volume": vol, "open_interest": oi,
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["spot"] = S0
        df.attrs["ticker"] = ticker
    return df


def surface_grid(df: pd.DataFrame, n_m: int = 30, n_t: int = 12):
    """
    Interpolate scattered (moneyness, T, IV) quotes onto a regular grid
    for 3D surface plotting. Linear interpolation with a nearest-neighbour
    fill for the convex-hull gaps.

    Returns (moneyness_grid, T_grid, IV_grid) or (None, None, None) if
    there is not enough data to interpolate.
    """
    from scipy.interpolate import griddata

    if df.empty or len(df) < 8 or df["T"].nunique() < 2:
        return None, None, None

    m_lin = np.linspace(df["moneyness"].min(), df["moneyness"].max(), n_m)
    t_lin = np.linspace(df["T"].min(), df["T"].max(), n_t)
    MM, TT = np.meshgrid(m_lin, t_lin)

    pts  = df[["moneyness", "T"]].values
    vals = df["implied_vol"].values

    IV = griddata(pts, vals, (MM, TT), method="linear")
    IV_near = griddata(pts, vals, (MM, TT), method="nearest")
    IV = np.where(np.isnan(IV), IV_near, IV)
    return MM, TT, IV


def skew_summary(df: pd.DataFrame) -> dict:
    """
    Quantify the skew for the nearest expiry: the IV difference between
    a 90% moneyness quote and a 110% one. Positive means downside
    protection is bid, the normal state of equity markets.
    """
    if df.empty:
        return {}
    near_T = df["T"].min()
    slice_ = df[np.isclose(df["T"], near_T)]
    if len(slice_) < 3:
        return {}

    def iv_at(target):
        idx = (slice_["moneyness"] - target).abs().idxmin()
        row = slice_.loc[idx]
        return row["implied_vol"], row["moneyness"]

    iv_low,  m_low  = iv_at(0.90)
    iv_atm,  m_atm  = iv_at(1.00)
    iv_high, m_high = iv_at(1.10)

    return {
        "expiry_T": near_T,
        "iv_90": iv_low, "m_90": m_low,
        "iv_atm": iv_atm, "m_atm": m_atm,
        "iv_110": iv_high, "m_110": m_high,
        "skew": iv_low - iv_high,
        "n_quotes": len(slice_),
    }
