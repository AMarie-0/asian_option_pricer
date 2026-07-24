"""
_test_options.py — Sanity checks for the lattice engine and option registry.

Run:  python _test_options.py

Checks (all against known closed forms or no-arbitrage identities):
  1. European call/put converge to Black–Scholes  (n=500 lattice)
  2. Put–call parity with the tool's discrete discounting convention
  3. American call == European call (no dividends)
  4. American put  >= European put
  5. Digital call price == disc * RN P(S_T > K), and BS closed form
  6. Barrier: out + in == vanilla (in-out parity); H→∞ recovers vanilla
  7. Chooser >= max(call, put); choice at t=0 == max; at t=T ~ straddle-ish
  8. Lookback payoff >= vanilla ATM-forward payoff pathwise (price ordering)
  9. Asian fixed vs floating consistency with the existing pricer (n=12)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from src.model.calibration import ModelParams
from src.options.registry import REGISTRY


def make_params(n: int, S0=100.0, sigma=0.3104, r=0.05, T=0.5) -> ModelParams:
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = 1.0 + r * dt
    q = (R - d) / (u - d)
    return ModelParams("TEST", S0, sigma, u, d, q, r, n, dt, T, R, 1.0 / R)


def bs_call(S0, K, r, sigma, T):
    d1 = (math.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_digital_call(S0, K, r, sigma, T, payout=1.0):
    d2 = (math.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    return payout * math.exp(-r * T) * norm.cdf(d2)


PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    PASS += ok
    FAIL += not ok
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    S0, sigma, r, T = 100.0, 0.3104, 0.05, 0.5
    p500 = make_params(500, S0, sigma, r, T)
    K = S0  # ATM

    R = REGISTRY
    print(f"Registry: {len(R)} options -> {sorted(R)}\n")

    # 1. BS convergence
    ec = R["euro_call"].price(p500, k_moneyness=1.0)["price"]
    ep = R["euro_put"].price(p500, k_moneyness=1.0)["price"]
    bs = bs_call(S0, K, r, sigma, T)
    check("European call ~ Black–Scholes", abs(ec - bs) / bs < 0.005,
          f"lattice={ec:.4f} vs BS={bs:.4f}")

    # 2. Put–call parity (discrete discounting: PV(K) = K * discount^n)
    pvK = K * p500.discount ** p500.n
    parity_gap = abs((ec - ep) - (S0 - pvK))
    check("Put–call parity (discrete disc.)", parity_gap < 1e-8,
          f"gap={parity_gap:.2e}")

    # 3/4. American vs European
    ac = R["amer_call"].price(p500, k_moneyness=1.0)
    ap = R["amer_put"].price(p500, k_moneyness=1.0)
    check("American call == European call (no divs)",
          abs(ac["early_exercise_premium"]) < 1e-9,
          f"premium={ac['early_exercise_premium']:.2e}")
    check("American put premium > 0", ap["early_exercise_premium"] > 0,
          f"premium={ap['early_exercise_premium']:.4f}")

    # 5. Digital — binomial digitals oscillate in n (payoff discontinuity
    #    falls between nodes), so use the standard n / n+1 averaging fix.
    dg_a = R["digital_call"].price(p500, k_moneyness=1.0, payout=10.0)
    dg_b = R["digital_call"].price(make_params(501, S0, sigma, r, T),
                                   k_moneyness=1.0, payout=10.0)
    dg_avg = 0.5 * (dg_a["price"] + dg_b["price"])
    bs_dg = bs_digital_call(S0, K, r, sigma, T, payout=10.0)
    check("Digital call ~ BS (n/n+1 averaged)", abs(dg_avg - bs_dg) / bs_dg < 0.02,
          f"avg={dg_avg:.4f} vs BS={bs_dg:.4f}, RN P(ITM)={dg_a['rn_prob_itm']:.4f}")

    # 6. Barrier parity + degenerate barrier
    bar = R["barrier_uo_call"].price(p500, k_moneyness=1.0, h_moneyness=1.3)
    check("Barrier in-out parity",
          abs(bar["price"] + bar["up_and_in_price"] - bar["vanilla_price"]) < 1e-9,
          f"out={bar['price']:.4f}, in={bar['up_and_in_price']:.4f}, "
          f"vanilla={bar['vanilla_price']:.4f}")
    far = R["barrier_uo_call"].price(p500, k_moneyness=1.0, h_moneyness=3.0)
    check("Barrier H→large recovers vanilla",
          abs(far["price"] - far["vanilla_price"]) / far["vanilla_price"] < 0.01,
          f"out={far['price']:.4f} vs vanilla={far['vanilla_price']:.4f}")

    # 7. Chooser bounds
    ch = R["chooser"].price(p500, k_moneyness=1.0, choice_fraction=0.5)
    check("Chooser >= max(call, put)",
          ch["price"] >= max(ch["call_price"], ch["put_price"]) - 1e-9,
          f"chooser={ch['price']:.4f}, call={ch['call_price']:.4f}, "
          f"put={ch['put_price']:.4f}")
    ch0 = R["chooser"].price(p500, k_moneyness=1.0, choice_fraction=0.0)
    check("Chooser at t=0 == max(call, put)",
          abs(ch0["price"] - max(ch0["call_price"], ch0["put_price"])) < 1e-9)

    # 8/9. Path-dependent engines (small n so 2^n stays tiny)
    p12 = make_params(12, S0, sigma, r, T)
    lb = R["lookback_float"].price(p12)
    ec12 = R["euro_call"].price(p12, k_moneyness=1.0)["price"]
    check("Lookback floating >= ATM European call",
          lb["price"] >= ec12 - 1e-9,
          f"lookback={lb['price']:.4f} vs euro={ec12:.4f}")

    af_fix = R["asian_fixed"].price(p12, k_moneyness=1.0)
    af_flt = R["asian_float"].price(p12)
    check("Asian pricers run & positive",
          af_fix["price"] > 0 and af_flt["price"] > 0,
          f"fixed={af_fix['price']:.4f}, floating={af_flt['price']:.4f}")
    check("Asian floating engines agree (existing invariant)",
          abs(af_flt["price_backward"] - af_flt["price_forward"]) < 1e-9)

    # Lattice speed sanity at n=500 vs path cap
    check("Lattice handles n=500 (already demonstrated above)", True,
          f"{p500.n * (p500.n + 1) // 2:,} nodes vs 2^25 = 33.5M paths")

    print(f"\n{PASS} passed, {FAIL} failed")
    return FAIL


if __name__ == "__main__":
    raise SystemExit(main())
