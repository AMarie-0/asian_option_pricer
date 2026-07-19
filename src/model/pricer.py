from __future__ import annotations

import numpy as np
from src.model.calibration import ModelParams
from src.model.binomial_tree import enumerate_paths, backward_induction


def price_asian_call(p: ModelParams) -> dict:
    """
    Price a European floating-strike Asian call option.

    Payoff: A_T = max(S_n - S_bar_n, 0)
    where   S_bar_n = C_n / (n+1),   C_n = sum_{t=0}^{n} S_t

    Two methods, both exact, must agree to floating-point precision:

    Method 1 — backward induction (V0_b):
        Fold the 2^n payoff vector backwards using the [up|down] split trick.

    Method 2 — direct forward calculation (V0_f):
        Recover up-move count j from terminal price algebraically,
        weight each path by q^j*(1-q)^(n-j), sum and discount.

    Returns a dict with all quantities needed by the UI and analytics.
    """
    # ── Forward enumeration ──────────────────────────────────────────
    S, C, w_phys = enumerate_paths(p)

    S_bar   = C / (p.n + 1)
    payoffs = np.maximum(S - S_bar, 0.0)

    # ── Method 1: backward induction ─────────────────────────────────
    V0_b = backward_induction(payoffs, p)

    # ── Method 2: direct forward calculation ─────────────────────────
    # S = S0 * u^j * d^(n-j)  =>  j = (log(S/S0) - n*log(d)) / log(u/d)
    log_ud = np.log(p.u / p.d)
    j = np.round((np.log(S / p.S0) - p.n * np.log(p.d)) / log_ud).astype(int)
    j = np.clip(j, 0, p.n)

    w_rn   = (p.q ** j) * ((1.0 - p.q) ** (p.n - j))
    disc_n = p.discount ** p.n
    E_Q    = float(np.sum(w_rn * payoffs))
    V0_f   = disc_n * E_Q

    # ── Physical-measure expected payoff (p=0.5) ─────────────────────
    # all w_phys are equal (= 0.5^n), so mean(payoffs) is exact
    E_P = float(np.mean(payoffs))

    return {
        # ── prices ──────────────────────────────────────────────────
        "price":              V0_b,          # canonical output
        "price_backward":     V0_b,
        "price_forward":      V0_f,
        # ── expected payoffs ─────────────────────────────────────────
        "expected_payoff_p":  E_P,           # under physical measure p=0.5
        "expected_payoff_q":  E_Q,           # under risk-neutral measure Q
        "discount_factor":    disc_n,
        # ── path arrays (for plots and analytics) ────────────────────
        "S_terminal":         S,
        "S_bar":              S_bar,
        "payoffs":            payoffs,
        "phys_weights":       w_phys,
        "rn_weights":         w_rn,
        "j_upmoves":          j,
        "n_paths":            len(S),
    }
