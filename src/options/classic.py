"""
classic.py — Five classic options on the recombining CRR lattice.

  1. European call          max(S_T - K, 0)
  2. European put           max(K - S_T, 0)
  3. American call          early exercise allowed (== European w/o dividends,
                            a useful teaching check the UI can display)
  4. American put           early exercise premium is real and visible
  5. Digital (cash-or-nothing) call    pays 1 if S_T > K else 0

All use the same calibrated ModelParams as the Asian pricer, and the same
per-step discounting convention (1 / (1 + r*dt)).

Strikes are entered as MONEYNESS (K as a fraction of S0) so the UI works
for any ticker without knowing the price level in advance.
"""
from __future__ import annotations

import numpy as np

from src.model.calibration import ModelParams
from src.model.lattice import price_lattice
from src.options.registry import ExtraInput, OptionSpec, register

_STRIKE_INPUT = ExtraInput(
    key="k_moneyness", label="Strike (× spot)", kind="moneyness",
    default=1.00, minimum=0.5, maximum=1.5,
    help="K = moneyness × S0. 1.00 = at-the-money.",
)


def _strike(p: ModelParams, k_moneyness: float) -> float:
    return float(k_moneyness) * p.S0


def european_call(p: ModelParams, k_moneyness: float = 1.0) -> dict:
    K = _strike(p, k_moneyness)
    res = price_lattice(p, lambda S: np.maximum(S - K, 0.0))
    res["strike"] = K
    return res


def european_put(p: ModelParams, k_moneyness: float = 1.0) -> dict:
    K = _strike(p, k_moneyness)
    res = price_lattice(p, lambda S: np.maximum(K - S, 0.0))
    res["strike"] = K
    return res


def american_call(p: ModelParams, k_moneyness: float = 1.0) -> dict:
    K = _strike(p, k_moneyness)
    payoff = lambda S: np.maximum(S - K, 0.0)          # noqa: E731
    res = price_lattice(p, payoff, american=True)
    res["strike"] = K
    res["european_price"] = price_lattice(p, payoff)["price"]
    res["early_exercise_premium"] = res["price"] - res["european_price"]
    return res


def american_put(p: ModelParams, k_moneyness: float = 1.0) -> dict:
    K = _strike(p, k_moneyness)
    payoff = lambda S: np.maximum(K - S, 0.0)          # noqa: E731
    res = price_lattice(p, payoff, american=True)
    res["strike"] = K
    res["european_price"] = price_lattice(p, payoff)["price"]
    res["early_exercise_premium"] = res["price"] - res["european_price"]
    return res


def digital_call(p: ModelParams, k_moneyness: float = 1.0, payout: float = 1.0) -> dict:
    """Cash-or-nothing call: pays `payout` (in currency units) iff S_T > K."""
    K = _strike(p, k_moneyness)
    res = price_lattice(p, lambda S: np.where(S > K, float(payout), 0.0))
    res["strike"] = K
    res["payout"] = float(payout)
    # price / discounted payout = risk-neutral P(S_T > K), a nice UI extra
    disc_T = p.discount ** p.n
    res["rn_prob_itm"] = res["price"] / (disc_T * payout) if payout else None
    return res


register(OptionSpec(
    key="euro_call", name="European Call", category="classic", engine="lattice",
    price=european_call, extra_inputs=(_STRIKE_INPUT,),
    description="Right to buy at strike K at maturity. Payoff max(S_T − K, 0).",
))
register(OptionSpec(
    key="euro_put", name="European Put", category="classic", engine="lattice",
    price=european_put, extra_inputs=(_STRIKE_INPUT,),
    description="Right to sell at strike K at maturity. Payoff max(K − S_T, 0).",
))
register(OptionSpec(
    key="amer_call", name="American Call", category="classic", engine="lattice",
    price=american_call, extra_inputs=(_STRIKE_INPUT,),
    description="Exercisable at any step. Without dividends its price equals "
                "the European call — the app can display this as a sanity check.",
))
register(OptionSpec(
    key="amer_put", name="American Put", category="classic", engine="lattice",
    price=american_put, extra_inputs=(_STRIKE_INPUT,),
    description="Exercisable at any step; carries a genuine early-exercise premium.",
))
register(OptionSpec(
    key="digital_call", name="Digital Call (cash-or-nothing)", category="classic",
    engine="lattice", price=digital_call,
    extra_inputs=(
        _STRIKE_INPUT,
        ExtraInput(key="payout", label="Cash payout", kind="float",
                   default=1.0, minimum=0.01, maximum=1000.0,
                   help="Amount paid if S_T finishes above K."),
    ),
    description="Pays a fixed cash amount iff S_T > K. Its discounted price "
                "reveals the risk-neutral probability of finishing in-the-money.",
))
