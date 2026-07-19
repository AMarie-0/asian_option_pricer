from __future__ import annotations

import math
import numpy as np
from src.model.calibration import ModelParams


def normal_approximation_price(p: ModelParams) -> dict:
    """
    Closed-form normal approximation of the floating-strike Asian call.
    Implements Section 7 of the report exactly.

    Key results:
      D_n = S_n - S_bar_n
      E[D_n] ≈ 0              (under p=0.5, small sigma^2*dt)
      Var(D_n) = S0^2 * sigma^2 * T * (2n+1) / (6*(n+1))

    Approximated price:
      V0_approx = e^{-rT} * v / sqrt(2*pi)   where v = sqrt(Var(D_n))

    Note: Section 7 explicitly switches to continuous time, so the
    discount here is e^{-rT}, NOT discount^n. This explains part of the
    gap with the exact binomial price (Section 7.3).

    Also returns the asymptotic limit as n -> infinity:
      V0_limit = e^{-rT} * S0 * sigma * sqrt(T/3) / sqrt(2*pi)
    """
    n   = p.n
    T   = p.T
    S0  = p.S0
    sig = p.sigma
    r   = p.r

    # Analytical variance of D_n = S_n - S_bar_n
    var_Dn = S0**2 * sig**2 * T * (2 * n + 1) / (6 * (n + 1))
    v      = math.sqrt(var_Dn)

    # Continuous discounting (Section 7 switches to continuous time)
    disc = math.exp(-r * T)

    price = disc * v / math.sqrt(2 * math.pi)

    # Asymptotic limit: (2n+1)/(n+1) -> 2, so Var -> S0^2*sigma^2*T/3
    price_limit = disc * S0 * sig * math.sqrt(T / 3.0) / math.sqrt(2 * math.pi)

    return {
        "var_Dn":      var_Dn,
        "sd_Dn":       v,
        "discount":    disc,
        "price":       price,
        "price_limit": price_limit,
    }


def empirical_Dn_stats(result: dict) -> dict:
    """
    Compute empirical mean and variance of D_n from the forward enumeration.
    Used to validate the analytical approximation (Table 4 in the report).

    result — output of price_asian_call()
    """
    D_n = result["S_terminal"] - result["S_bar"]
    w   = result["phys_weights"]

    emp_mean = float(np.average(D_n, weights=w))
    emp_var  = float(np.average((D_n - emp_mean) ** 2, weights=w))
    emp_sd   = math.sqrt(emp_var)

    return {
        "emp_mean": emp_mean,
        "emp_var":  emp_var,
        "emp_sd":   emp_sd,
    }
