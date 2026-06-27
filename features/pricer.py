# src/model/pricer.py
import numpy as np
from src.model.calibration import ModelParams
from src.model.binomial_tree import enumerate_paths
from src.model.payoff import asian_call_payoff

def price_asian_call(p: ModelParams) -> dict:
    """
    Compute risk-neutral price, expected payoff, and full
    terminal distribution for analysis.
    """
    states  = enumerate_paths(p)
    payoffs = [asian_call_payoff(s, p) for s in states]
    weights = [s.weight for s in states]

    # Discount factor
    total_r   = (1 + p.r) ** (252 * p.T) - 1
    discount  = 1 / (1 + total_r)

    expected_payoff = sum(w * v for w, v in zip(weights, payoffs))
    price           = discount * expected_payoff

    return {
        "price":            price,
        "expected_payoff":  expected_payoff,
        "discount":         discount,
        "terminal_states":  states,
        "payoffs":          payoffs,
        "weights":          weights,
        "n_states":         len(states),
    }
