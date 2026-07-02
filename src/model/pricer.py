from __future__ import annotations

import math
import numpy as np
from src.model.calibration import ModelParams
from src.model.binomial_tree import enumerate_paths
from src.model.payoff import asian_call_payoff


def price_asian_call(p: ModelParams) -> dict:
    """Compute risk-neutral price and full terminal distribution."""
    states  = enumerate_paths(p)
    payoffs = np.array([asian_call_payoff(s, p) for s in states])
    weights = np.array([s.weight              for s in states])

    # continuous discounting — r is annual, T is in years
    discount        = math.exp(-p.r * p.T)
    expected_payoff = float(np.dot(weights, payoffs))
    price           = discount * expected_payoff

    return {
        "price":           price,
        "expected_payoff": expected_payoff,
        "discount":        discount,
        "terminal_states": states,
        "payoffs":         payoffs,
        "weights":         weights,
        "n_states":        len(states),
    }
