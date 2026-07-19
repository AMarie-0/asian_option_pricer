from __future__ import annotations

import numpy as np
from src.model.calibration import ModelParams


def enumerate_paths(p: ModelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward enumeration of all 2^n paths using numpy arrays.

    Mirrors the R code exactly:
      - Start with (S=S0, C=S0, w=1.0)
      - At each step branch into [up | down] concatenated in that order
      - After n steps: 2^n rows, one per unique path

    Returns
    -------
    S : (2^n,)  terminal stock price per path
    C : (2^n,)  cumulative price sum along each path (incl. S0 and Sn)
    w : (2^n,)  physical probability weight (all equal to 0.5^n)
    """
    S = np.array([p.S0], dtype=np.float64)
    C = np.array([p.S0], dtype=np.float64)
    w = np.array([1.0],  dtype=np.float64)

    for _ in range(p.n):
        S_up = S * p.u
        S_dn = S * p.d
        C = np.concatenate([C + S_up, C + S_dn])
        S = np.concatenate([S_up, S_dn])
        w = np.concatenate([w * 0.5, w * 0.5])

    return S, C, w


def backward_induction(payoffs: np.ndarray, p: ModelParams) -> float:
    """
    R-style backward induction on the 2^n payoff vector.

    The [up | down] concatenation order in enumerate_paths guarantees that
    at every backward step the first half are up-children and the second
    half are down-children of the same parents.  We exploit this to fold
    the vector in half n times with pure vectorised arithmetic — no tree
    traversal, no dictionary, no rounding artefacts.

    Equivalent to: V0 = discount^n * E^Q[A_T]
    """
    values = payoffs.copy()
    for _ in range(p.n):
        half   = len(values) // 2
        v_up   = values[:half]
        v_dn   = values[half:]
        values = p.discount * (p.q * v_up + (1.0 - p.q) * v_dn)
    return float(values[0])


def build_stock_matrix(p: ModelParams) -> np.ndarray:
    """
    (n+1)×(n+1) price matrix for the binomial tree visualisation.
    Entry [i, j] = S0 * u^j * d^(i-j), upper-triangular (NaN elsewhere).
    """
    mat = np.full((p.n + 1, p.n + 1), np.nan)
    for i in range(p.n + 1):
        for j in range(i + 1):
            mat[i, j] = p.S0 * (p.u ** j) * (p.d ** (i - j))
    return mat
