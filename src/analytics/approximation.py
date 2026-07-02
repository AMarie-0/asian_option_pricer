from __future__ import annotations

import math
from src.model.calibration import ModelParams


def normal_approximation_price(p: ModelParams) -> float:
    """
    Closed-form normal approximation:
      V0 = e^{-rT} * S0 * sigma * sqrt(T/6 * (2n+1)/(n+1)) / sqrt(2*pi)
    """
    n   = p.n
    fac = math.sqrt((p.T / 6) * (2 * n + 1) / (n + 1))
    v   = p.S0 * p.sigma * fac

    total_r  = (1 + p.r) ** (252 * p.T) - 1
    discount = 1 / (1 + total_r)

    return discount * v / math.sqrt(2 * math.pi)
