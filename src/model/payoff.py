from __future__ import annotations

from src.model.binomial_tree import PathState
from src.model.calibration import ModelParams


def asian_call_payoff(state: PathState, p: ModelParams) -> float:
    """Floating-strike Asian call: max(S_n - avg(S), 0)"""
    avg = state.cum_sum / (p.n + 1)
    return max(state.price - avg, 0.0)
