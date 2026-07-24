"""
exotic.py — Five exotic options.

Path-dependent (reuse the existing 2^n path engine, n capped like today):
  1. Asian call, floating strike   — delegates to the EXISTING pricer untouched
  2. Asian call, fixed strike      — same enumeration, payoff max(S̄ − K, 0)
  3. Lookback call, floating strike— payoff S_T − min(S_t), needs running min

Lattice-friendly (recombining, O(n^2), n can be large):
  4. Up-and-out barrier call       — knocked out if S ever ≥ H
  5. Chooser option                — at step m, holder picks call or put

The path-dependent pricers deliberately mirror binomial_tree.enumerate_paths'
[up | down] concatenation so backward_induction keeps working unchanged.
"""
from __future__ import annotations

import numpy as np

from src.model.binomial_tree import backward_induction, enumerate_paths
from src.model.calibration import ModelParams
from src.model.lattice import price_lattice, spot_at_step
from src.model.pricer import price_asian_call
from src.options.registry import ExtraInput, OptionSpec, register

_STRIKE_INPUT = ExtraInput(
    key="k_moneyness", label="Strike (× spot)", kind="moneyness",
    default=1.00, minimum=0.5, maximum=1.5,
    help="K = moneyness × S0.",
)


# ── 1. Asian floating strike — the existing tool, untouched ──────────────────

def asian_floating_call(p: ModelParams) -> dict:
    """Thin wrapper so the registry exposes the existing pricer as-is."""
    return price_asian_call(p)


# ── 2. Asian fixed strike ────────────────────────────────────────────────────

def asian_fixed_call(p: ModelParams, k_moneyness: float = 1.0) -> dict:
    """
    Payoff max(S̄_n − K, 0) with S̄_n the arithmetic average incl. S0.
    Reuses enumerate_paths: C already holds the cumulative sum per path.
    """
    K = float(k_moneyness) * p.S0
    S, C, w_phys = enumerate_paths(p)
    S_bar = C / (p.n + 1)
    payoffs = np.maximum(S_bar - K, 0.0)
    price = backward_induction(payoffs, p)
    return {
        "price": price, "strike": K,
        "S_terminal": S, "S_bar": S_bar,
        "payoffs": payoffs, "phys_weights": w_phys, "n_paths": len(S),
    }


# ── 3. Lookback floating strike ──────────────────────────────────────────────

def lookback_floating_call(p: ModelParams) -> dict:
    """
    Payoff S_T − min_{0≤t≤n} S_t  (always ≥ 0, no max() needed).
    Same forward enumeration as the Asian engine, but tracking the running
    MINIMUM per path instead of the cumulative sum. The [up | down]
    concatenation order is preserved so backward_induction applies directly.
    """
    S = np.array([p.S0], dtype=np.float64)
    M = np.array([p.S0], dtype=np.float64)      # running minimum
    for _ in range(p.n):
        S_up, S_dn = S * p.u, S * p.d
        M = np.concatenate([np.minimum(M, S_up), np.minimum(M, S_dn)])
        S = np.concatenate([S_up, S_dn])

    payoffs = S - M
    price = backward_induction(payoffs, p)
    return {
        "price": price,
        "S_terminal": S, "S_min": M,
        "payoffs": payoffs, "n_paths": len(S),
    }


# ── 4. Up-and-out barrier call (lattice) ─────────────────────────────────────

def barrier_up_out_call(
    p: ModelParams, k_moneyness: float = 1.0,
    h_moneyness: float = 1.3, rebate: float = 0.0,
) -> dict:
    """
    Vanilla call knocked out (worthless, or paying a rebate) the moment the
    lattice node price reaches H = h_moneyness × S0.

    Note: barrier monitoring is at lattice nodes, so the price converges to
    the continuously-monitored value only as n grows — one more reason the
    lattice engine's large-n headroom matters.
    Also returns the up-and-in price via in-out parity.
    """
    K = float(k_moneyness) * p.S0
    H = float(h_moneyness) * p.S0
    if H <= p.S0:
        raise ValueError("Up-and-out barrier must be above the spot price.")
    payoff = lambda S: np.maximum(S - K, 0.0)          # noqa: E731

    res = price_lattice(p, payoff, barrier=H,
                        barrier_type="up-and-out", rebate=rebate)
    vanilla = price_lattice(p, payoff)["price"]
    res.update({
        "strike": K, "barrier": H, "rebate": float(rebate),
        "vanilla_price": vanilla,
        "up_and_in_price": vanilla - res["price"] + 0.0 if rebate == 0.0 else None,
    })
    return res


# ── 5. Chooser option (lattice) ──────────────────────────────────────────────

def chooser(p: ModelParams, k_moneyness: float = 1.0,
            choice_fraction: float = 0.5) -> dict:
    """
    Simple chooser: at step m = round(choice_fraction × n) the holder picks
    whichever of the K-strike European call or put is worth more, then holds
    it to maturity.

    Priced by running two backward inductions to step m (call and put
    continuation values via price_lattice's snapshot), taking the nodewise
    max, then folding the remaining m steps back to the root.
    """
    K = float(k_moneyness) * p.S0
    m = int(round(float(choice_fraction) * p.n))
    m = max(0, min(p.n, m))

    call = price_lattice(p, lambda S: np.maximum(S - K, 0.0), snapshot_step=m)
    put  = price_lattice(p, lambda S: np.maximum(K - S, 0.0), snapshot_step=m)

    values = np.maximum(call["snapshot_values"], put["snapshot_values"])
    for _ in range(m):
        values = p.discount * (p.q * values[1:] + (1.0 - p.q) * values[:-1])

    return {
        "price": float(values[0]),
        "strike": K, "choice_step": m,
        "call_price": call["price"], "put_price": put["price"],
        "S_terminal": spot_at_step(p, p.n),
    }


register(OptionSpec(
    key="asian_float", name="Asian Call (floating strike)", category="exotic",
    engine="paths", price=asian_floating_call,
    description="The original tool: payoff max(S_T − S̄, 0) on the average "
                "including S0. Exact via full path enumeration.",
))
register(OptionSpec(
    key="asian_fixed", name="Asian Call (fixed strike)", category="exotic",
    engine="paths", price=asian_fixed_call, extra_inputs=(_STRIKE_INPUT,),
    description="Payoff max(S̄ − K, 0). Same enumeration machinery, K chosen "
                "as moneyness × S0.",
))
register(OptionSpec(
    key="lookback_float", name="Lookback Call (floating strike)",
    category="exotic", engine="paths", price=lookback_floating_call,
    description="Payoff S_T − min S_t: buy at the lowest price seen. "
                "Tracks the running minimum along every path.",
))
register(OptionSpec(
    key="barrier_uo_call", name="Barrier Call (up-and-out)", category="exotic",
    engine="lattice", price=barrier_up_out_call,
    extra_inputs=(
        _STRIKE_INPUT,
        ExtraInput(key="h_moneyness", label="Barrier (× spot)", kind="moneyness",
                   default=1.30, minimum=1.01, maximum=3.0,
                   help="Knock-out level H = moneyness × S0, must exceed 1."),
        ExtraInput(key="rebate", label="Rebate", kind="float",
                   default=0.0, minimum=0.0, maximum=100.0,
                   help="Cash paid if knocked out."),
    ),
    description="Vanilla call that dies if the price ever touches H. "
                "Up-and-in twin recovered by in-out parity.",
))
register(OptionSpec(
    key="chooser", name="Chooser Option", category="exotic", engine="lattice",
    price=chooser,
    extra_inputs=(
        _STRIKE_INPUT,
        ExtraInput(key="choice_fraction", label="Choice date (fraction of T)",
                   kind="moneyness", default=0.5, minimum=0.05, maximum=0.95,
                   help="When the holder must choose call vs put."),
    ),
    description="At an intermediate date the holder picks call or put — a "
                "straddle-like bet on volatility, cheaper than buying both.",
))
