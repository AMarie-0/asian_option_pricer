from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace
from src.model.calibration import ModelParams


@dataclass
class PathState:
    price:   float
    cum_sum: float
    weight:  float


def build_stock_tree(p: ModelParams) -> np.ndarray:
    """Return (n+1)×(n+1) matrix where tree[i,j] = S0 * u^j * d^(i-j)."""
    tree = np.zeros((p.n + 1, p.n + 1))
    for i in range(p.n + 1):
        for j in range(i + 1):
            tree[i, j] = p.S0 * (p.u ** j) * (p.d ** (i - j))
    return tree


def enumerate_paths(p: ModelParams) -> list[PathState]:
    """
    Forward induction tracking (price, cumulative_sum, weight).
    Merges states with identical (price, cum_sum) for efficiency.
    """
    current: dict[tuple, PathState] = {
        (round(p.S0, 8), round(p.S0, 8)): PathState(p.S0, p.S0, 1.0)
    }

    for _ in range(p.n):
        nxt: dict[tuple, PathState] = {}
        for state in current.values():
            for factor, prob in [(p.u, p.q), (p.d, 1 - p.q)]:
                new_price  = state.price   * factor
                new_cum    = state.cum_sum + new_price
                new_weight = state.weight  * prob
                key = (round(new_price, 6), round(new_cum, 6))
                if key in nxt:
                    nxt[key].weight += new_weight
                else:
                    nxt[key] = PathState(new_price, new_cum, new_weight)
        current = nxt

    return list(current.values())
