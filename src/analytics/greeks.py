"""
greeks.py — Option sensitivities (Greeks).

Two approaches, both provided so the UI can show them converging:

1. FINITE DIFFERENCE (works for EVERY option in the registry)
   Bump an input, re-price, divide by the bump. Model-agnostic: it does not
   care whether the pricer is a lattice, a path enumeration, or Monte Carlo.

2. ANALYTICAL BLACK-SCHOLES (European call/put only)
   The closed forms the course derives from the BS PDE. Used as the exact
   reference the binomial Greeks should converge to as n grows.

Bump sizes
----------
Central differences are used wherever possible (error O(h^2) rather than O(h)).
The spot bump is multiplicative (h = 1% of S0) so it scales across tickers
priced at $5 or $500. Gamma needs the second difference, so it reuses the
same three prices as delta — no extra pricing calls.

Cost note: a full Greek set costs ~7 re-pricings. On the O(n^2) lattice this
is instant. On the O(2^n) path engine at n=25 each pricing takes ~10s, so the
UI should compute path-engine Greeks at reduced n, or warn the user.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.stats import norm

from src.model.calibration import ModelParams


def _rebuild(p: ModelParams, *, S0=None, sigma=None, T=None, r=None) -> ModelParams:
    """
    Return a copy of p with one input changed, re-deriving every dependent
    quantity (u, d, q, dt, discount) from the CRR relations.

    This matters: naively bumping sigma without recomputing u and d would
    measure nothing, since the tree geometry IS the volatility.
    """
    S0    = p.S0    if S0    is None else S0
    sigma = p.sigma if sigma is None else sigma
    T     = p.T     if T     is None else T
    r     = p.r     if r     is None else r

    dt = T / p.n
    u  = math.exp(sigma * math.sqrt(dt))
    d  = 1.0 / u
    R  = 1.0 + r * dt
    q  = (R - d) / (u - d)

    return dataclasses.replace(
        p, S0=S0, sigma=sigma, T=T, r=r,
        dt=dt, u=u, d=d, R_period=R, q=q, discount=1.0 / R,
    )


def _rescale_moneyness(extra: dict, S0_new: float, S0_ref: float) -> dict:
    """
    CRITICAL for correct delta/gamma.

    Strike-like inputs in this tool are expressed as MONEYNESS (K = m * S0),
    so naively bumping S0 drags the strike along with it and measures a
    pure scaling effect (delta ~ V/S) instead of the true dV/dS.

    Delta means: how does the value of a contract with a FIXED dollar strike
    change when spot moves? So when we bump S0 we must rescale the moneyness
    inputs to keep the dollar strike/barrier constant:

        K = m_old * S0_ref  ==>  m_new = m_old * S0_ref / S0_new
    """
    if S0_new <= 0:
        return dict(extra)
    ratio = S0_ref / S0_new
    out = dict(extra)
    for key in ("k_moneyness", "h_moneyness"):
        if key in out:
            out[key] = out[key] * ratio

    # An up-barrier must stay strictly above spot or the pricer rejects it.
    # Bumping spot UP shrinks h_moneyness towards 1.0, and a barrier close to
    # spot can cross below it — a validation artefact of the bump, not a real
    # input error. Clamp just above 1 so the finite difference still evaluates.
    if "h_moneyness" in out:
        out["h_moneyness"] = max(out["h_moneyness"], 1.0001)
    return out


def compute_greeks(
    p: ModelParams,
    price_fn,
    extra_inputs: dict | None = None,
    bump_spot: float = 0.01,     # 1% of S0
    bump_vol: float = 0.01,      # 1 vol point (absolute)
    bump_rate: float = 0.0001,   # 1 basis point
    bump_days: float = 1.0,      # 1 calendar day
) -> dict:
    """
    Finite-difference Greeks for any pricer with signature f(params, **inputs).

    Returns a dict of Greeks in *market* conventions:
      delta  — dV/dS            (per $1 of spot)
      gamma  — d2V/dS2          (delta change per $1 of spot)
      vega   — dV/dsigma        (per 1 percentage point of vol)
      theta  — dV/dt            (per CALENDAR DAY, negative = time decay)
      rho    — dV/dr            (per 1 percentage point of rate)
    """
    extra = extra_inputs or {}
    f = lambda pp: price_fn(pp, **extra)["price"]   # noqa: E731

    def price_at_spot(S_new):
        """Re-price at a new spot, holding the DOLLAR strike fixed."""
        return price_fn(_rebuild(p, S0=S_new),
                        **_rescale_moneyness(extra, S_new, p.S0))["price"]

    v_mid = f(p)

    # ── Delta: central difference, small bump is fine (first-order) ─────
    h = p.S0 * bump_spot
    delta = (price_at_spot(p.S0 + h) - price_at_spot(p.S0 - h)) / (2 * h)

    # ── Gamma: needs a WIDER bump than delta ────────────────────────────
    # The lattice prices a piecewise-linear function of spot: between two
    # adjacent nodes the value is essentially linear, so a second difference
    # taken inside one cell reads near-zero curvature, and one straddling a
    # single node reads a huge spike. Either way the sawtooth swamps the
    # true gamma. Averaging over several node-widths recovers the smooth
    # second derivative. Node spacing near S0 is S0*(u-1), so we take the
    # larger of the requested bump and ~4 node widths.
    node_spacing = p.S0 * (p.u - 1.0)
    h_g = max(h, 4.0 * node_spacing, p.S0 * 0.02)
    gamma = (price_at_spot(p.S0 + h_g) - 2 * v_mid
             + price_at_spot(p.S0 - h_g)) / (h_g ** 2)

    # ── Vega: central difference in sigma, reported per vol POINT ───────
    v_vol_up   = f(_rebuild(p, sigma=p.sigma + bump_vol))
    v_vol_down = f(_rebuild(p, sigma=max(p.sigma - bump_vol, 1e-6)))
    vega = (v_vol_up - v_vol_down) / (2 * bump_vol) * 0.01

    # ── Theta: forward difference (T can't go up past the contract) ─────
    dT = bump_days / 365.0
    if p.T - dT > 1e-6:
        v_theta = f(_rebuild(p, T=p.T - dT))
        theta = (v_theta - v_mid) / bump_days   # per calendar day
    else:
        theta = float("nan")

    # ── Rho: central difference in r, reported per rate POINT ───────────
    v_r_up   = f(_rebuild(p, r=p.r + bump_rate))
    v_r_down = f(_rebuild(p, r=max(p.r - bump_rate, 0.0)))
    rho = (v_r_up - v_r_down) / (2 * bump_rate) * 0.01

    return {
        "price": v_mid,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
        "rho":   rho,
        "n_pricings": 7,
    }


def black_scholes_greeks(p: ModelParams, K: float, kind: str = "call") -> dict:
    """
    Closed-form Black-Scholes Greeks — the continuous-time limit the
    binomial Greeks converge to. Derived from the BS PDE:

        dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + r S dV/dS - rV = 0

    Conventions match compute_greeks: vega and rho per percentage point,
    theta per calendar day.
    """
    S0, sig, r, T = p.S0, p.sigma, p.r, p.T
    sqT = sig * math.sqrt(T)
    d1  = (math.log(S0 / K) + (r + sig ** 2 / 2) * T) / sqT
    d2  = d1 - sqT
    disc = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sqT)
    vega  = S0 * pdf_d1 * math.sqrt(T) * 0.01

    if kind == "call":
        price = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S0 * pdf_d1 * sig / (2 * math.sqrt(T))
                 - r * K * disc * norm.cdf(d2)) / 365.0
        rho   = K * T * disc * norm.cdf(d2) * 0.01
    else:
        price = K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (-S0 * pdf_d1 * sig / (2 * math.sqrt(T))
                 + r * K * disc * norm.cdf(-d2)) / 365.0
        rho   = -K * T * disc * norm.cdf(-d2) * 0.01

    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


def greeks_profile(
    p: ModelParams, price_fn, extra_inputs: dict | None = None,
    n_points: int = 25, spot_range: float = 0.4,
) -> dict:
    """
    Delta and gamma across a range of spot prices — the classic
    "delta vs spot" S-curve and the gamma bell centred at the strike.

    Costs 3 pricings per grid point, so keep n_points modest on the
    path engine.
    """
    extra = extra_inputs or {}
    f = lambda pp: price_fn(pp, **extra)["price"]   # noqa: E731

    spots  = np.linspace(p.S0 * (1 - spot_range), p.S0 * (1 + spot_range), n_points)
    deltas, gammas, prices = [], [], []

    # Strike stays fixed in dollars across the whole sweep, anchored at the
    # ORIGINAL spot — otherwise every grid point would silently re-strike
    # the option ATM and delta would be flat.
    for S in spots:
        h = S * 0.01
        g = lambda SS: price_fn(_rebuild(p, S0=SS),                    # noqa: E731
                                **_rescale_moneyness(extra, SS, p.S0))["price"]
        vu, vm, vd = g(S + h), g(S), g(S - h)
        prices.append(vm)
        deltas.append((vu - vd) / (2 * h))
        gammas.append((vu - 2 * vm + vd) / (h ** 2))

    return {
        "spots":  spots,
        "prices": np.array(prices),
        "deltas": np.array(deltas),
        "gammas": np.array(gammas),
    }
