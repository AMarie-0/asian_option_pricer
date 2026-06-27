# src/model/binomial_tree.py
import numpy as np
from dataclasses import dataclass, field
from src.model.calibration import ModelParams

@dataclass
class PathState:
    price:  float
    cum_sum: float
    weight: float   # probability weight

def build_stock_tree(p: ModelParams) -> np.ndarray:
    """
    Return (n+1)×(n+1) matrix where tree[i,j] = S0 * u^j * d^(i-j).
    Entry is 0 where j > i (invalid).
    """
    tree = np.zeros((p.n + 1, p.n + 1))
    for i in range(p.n + 1):
        for j in range(i + 1):
            tree[i, j] = p.S0 * (p.u ** j) * (p.d ** (i - j))
    return tree

def enumerate_paths(p: ModelParams) -> list[PathState]:
    """
    Forward induction: track (price, cumulative_sum, weight).
    Returns list of terminal PathStates (one per distinct path).
    Memory-efficient: prune by merging states with same price only
    if recombination is enabled — here we keep all 2^n paths for
    exact Asian payoff computation.
    """
    # Use dict {(price_rounded, cum_sum_rounded): PathState}
    # for merging equivalent states efficiently
    current: dict[tuple, PathState] = {
        (round(p.S0, 8), round(p.S0, 8)): PathState(p.S0, p.S0, 1.0)
    }

    for _ in range(p.n):
        nxt: dict[tuple, PathState] = {}
        for state in current.values():
            for factor, prob in [(p.u, p.q), (p.d, 1 - p.q)]:
                new_price   = state.price   * factor
                new_cum     = state.cum_sum + new_price
                new_weight  = state.weight  * prob
                key = (round(new_price, 6), round(new_cum, 6))
                if key in nxt:
                    nxt[key].weight += new_weight
                else:
                    nxt[key] = PathState(new_price, new_cum, new_weight)
        current = nxt

    return list(current.values())
