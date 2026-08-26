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


def _quotes_via_yfinance(ticker: str, max_expiries: int, kind: str):
    """Primary source: yfinance option_chain. Returns (S0, [raw quote dicts])."""
    import yfinance as yf
    tk = yf.Ticker(ticker)

    expiries = tk.options
    if not expiries:
        return None, []

    S0 = None
    try:
        S0 = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
    except Exception:
        pass
    if not S0:
        S0 = float(tk.history(period="5d")["Close"].iloc[-1])
    S0 = float(S0)

    today = datetime.now().date()
    raw = []
    for exp in expiries[:max_expiries]:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            T = (exp_date - today).days / 365.0
            if T <= 0.01:
                continue
            chain = tk.option_chain(exp)
            quotes = chain.calls if kind == "call" else chain.puts
        except Exception:
            continue
        for _, q in quotes.iterrows():
            raw.append({
                "expiry": exp, "T": T,
                "strike": float(q.get("strike", 0) or 0),
                "bid": float(q.get("bid", 0) or 0),
                "ask": float(q.get("ask", 0) or 0),
                "last": float(q.get("lastPrice", 0) or 0),
                "volume": int(q.get("volume", 0) or 0),
                "oi": int(q.get("openInterest", 0) or 0),
            })
    return S0, raw


def _quotes_via_requests(ticker: str, max_expiries: int, kind: str):
    """
    Fallback source: Yahoo's options JSON endpoint hit directly with a
    browser User-Agent — the same trick data/fetch.py uses for prices when
    yfinance is blocked from datacenter IPs (e.g. Streamlit Cloud).

    GET /v7/finance/options/{ticker}            -> expirationDates + spot
    GET /v7/finance/options/{ticker}?date=EPOCH -> chain for one expiry
    """
    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    sess = requests.Session()
    sess.headers.update(headers)

    def get_json(url):
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("optionChain", {}).get("result", [])
        return result[0] if result else None

    base = None
    for host in ("query1", "query2"):
        try:
            base_url = f"https://{host}.finance.yahoo.com/v7/finance/options/{ticker}"
            base = get_json(base_url)
            if base:
                break
        except Exception:
            continue
    if not base:
        return None, []

    S0 = float(base.get("quote", {}).get("regularMarketPrice") or 0)
    if S0 <= 0:
        return None, []

    exp_epochs = base.get("expirationDates", [])[:max_expiries]
    today = datetime.now().date()
    raw = []

    for i, epoch in enumerate(exp_epochs):
        try:
            if i == 0 and base.get("options"):
                block = base["options"][0]        # first chain rides along
            else:
                block = get_json(f"{base_url}?date={epoch}")
                block = block["options"][0] if block and block.get("options") else None
            if not block:
                continue
            exp_date = datetime.utcfromtimestamp(epoch).date()
            exp = exp_date.strftime("%Y-%m-%d")
            T = (exp_date - today).days / 365.0
            if T <= 0.01:
                continue
            for q in block.get("calls" if kind == "call" else "puts", []):
                raw.append({
                    "expiry": exp, "T": T,
                    "strike": float(q.get("strike", 0) or 0),
                    "bid": float(q.get("bid", 0) or 0),
                    "ask": float(q.get("ask", 0) or 0),
                    "last": float(q.get("lastPrice", 0) or 0),
                    "volume": int(q.get("volume", 0) or 0),
                    "oi": int(q.get("openInterest", 0) or 0),
                })
        except Exception:
            continue
    return S0, raw


def build_surface_from_quotes(S0, raw_quotes, r=0.04,
                              moneyness_range=(0.7, 1.3),
                              min_volume=1, kind="call") -> pd.DataFrame:
    """
    Shared filtering + IV inversion over raw quote dicts, whichever source
    produced them.

    Off-hours handling: when the market is closed Yahoo zeroes bids/asks,
    which used to filter out EVERY quote. If bid/ask are unusable but a
    last-traded price exists on a quote with real open interest, we fall
    back to that price and flag the quote as stale. The resulting frame
    carries attrs["stale_quotes"] so the UI can caveat the surface.
    """
    rows, stale_count = [], 0
    for q in raw_quotes:
        try:
            K = q["strike"]
            if K <= 0:
                continue
            m = K / S0
            if not (moneyness_range[0] <= m <= moneyness_range[1]):
                continue

            bid, ask = q["bid"], q["ask"]
            has_market = bid > 0 and ask >= bid
            if has_market:
                if (ask - bid) / ((ask + bid) / 2) > 0.6:
                    continue
                if q["volume"] < min_volume and q["oi"] < 10:
                    continue
                mid, stale = (bid + ask) / 2, False
            else:
                # Market closed / quotes zeroed: last trade as best estimate,
                # gated on open interest so dead strikes stay out.
                if q["last"] <= 0 or q["oi"] < 10:
                    continue
                mid, stale = q["last"], True

            iv = implied_vol(mid, S0, K, r, q["T"], kind)
            if not np.isfinite(iv) or iv < 0.01 or iv > 3.0:
                continue

            stale_count += stale
            rows.append({
                "expiry": q["expiry"], "T": q["T"], "strike": K,
                "moneyness": m, "mid_price": mid, "implied_vol": iv,
                "volume": q["volume"], "open_interest": q["oi"],
                "stale": stale,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["spot"] = S0
        df.attrs["stale_quotes"] = int(stale_count)
    return df


def fetch_vol_surface(
    ticker: str, r: float = 0.04, max_expiries: int = 8,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
    min_volume: int = 1, kind: str = "call",
) -> pd.DataFrame:
    """
    Build the implied-vol surface for `ticker` from option chains.

    Source order:
      1. yfinance (works locally)
      2. direct HTTP to Yahoo's options endpoint (survives datacenter-IP
         blocking on Streamlit Cloud, mirroring data/fetch.py)

    Off-hours, zero-bid quotes fall back to last-traded prices and are
    flagged via attrs["stale_quotes"].

    An empty frame means no usable quotes from either source.
    """
    S0, raw = None, []
    try:
        S0, raw = _quotes_via_yfinance(ticker, max_expiries, kind)
    except Exception:
        pass
    if not raw:
        try:
            S0, raw = _quotes_via_requests(ticker, max_expiries, kind)
        except Exception:
            pass
    if not raw or not S0:
        return pd.DataFrame()

    df = build_surface_from_quotes(S0, raw, r, moneyness_range, min_volume, kind)
    if not df.empty:
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