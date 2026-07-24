"""
lattice.py — Recombining CRR binomial lattice engine.

Why a second engine?
--------------------
The existing binomial_tree.py enumerates all 2^n paths, which is REQUIRED
for path-dependent payoffs (Asian averages, lookback extrema) but wasteful
for everything else. Options whose value at a node depends only on the
spot price at that node (European, American, digital, barrier, chooser)
recombine: the tree has only n+1 terminal nodes and n(n+1)/2 total nodes.

  Path enumeration : O(2^n)  -> n capped at ~25   (33M paths)
  Recombining      : O(n^2)  -> n = 500+ is instant

Both engines share the same ModelParams from calibration.py, so the
existing data pipeline (fetch -> calibrate) feeds either one unchanged.

Node convention
---------------
At step i (0..n), node j (0..i) has price  S0 * u^j * d^(i-j).
Arrays are indexed by j, ascending (lowest price first).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.model.calibration import ModelParams


def spot_at_step(p: ModelParams, i: int) -> np.ndarray:
    """Vector of the i+1 node prices at step i, ascending in j."""
    j = np.arange(i + 1)
    return p.S0 * (p.u ** j) * (p.d ** (i - j))


def price_lattice(
    p: ModelParams,
    terminal_payoff: Callable[[np.ndarray], np.ndarray],
    american: bool = False,
    intrinsic: Callable[[np.ndarray], np.ndarray] | None = None,
    barrier: float | None = None,
    barrier_type: str | None = None,   # "up-and-out" | "down-and-out"
    rebate: float = 0.0,
    snapshot_step: int | None = None,
) -> dict:
    """
    Generic backward induction on the recombining lattice.

    Parameters
    ----------
    terminal_payoff : f(S_terminal) -> payoffs, vectorised
    american        : if True, allow early exercise using `intrinsic`
                      (defaults to terminal_payoff if not given)
    barrier         : knock-out level H (None = no barrier)
    barrier_type    : "up-and-out"  -> value = rebate where S >= H
                      "down-and-out"-> value = rebate where S <= H
                      Knock-IN options are derived via in-out parity by the
                      caller (V_in = V_vanilla - V_out).
    snapshot_step   : if set, also return the continuation values observed
                      at that step (used by the chooser option).

    Uses the SAME discounting convention as the path engine:
    one-step discount = p.discount = 1 / (1 + r*dt).

    Returns dict with "price", plus terminal arrays for plotting.
    """
    if american and intrinsic is None:
        intrinsic = terminal_payoff

    S_T = spot_at_step(p, p.n)
    values = terminal_payoff(S_T).astype(np.float64)

    def apply_barrier(vals: np.ndarray, S: np.ndarray) -> np.ndarray:
        if barrier is None:
            return vals
        if barrier_type == "up-and-out":
            return np.where(S >= barrier, rebate, vals)
        if barrier_type == "down-and-out":
            return np.where(S <= barrier, rebate, vals)
        raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

    values = apply_barrier(values, S_T)
    early_exercise_nodes = 0
    snapshot = None

    for i in range(p.n - 1, -1, -1):
        # node j at step i has children j (down) and j+1 (up) at step i+1
        values = p.discount * (p.q * values[1:] + (1.0 - p.q) * values[:-1])
        S_i = spot_at_step(p, i)

        if snapshot_step is not None and i == snapshot_step:
            snapshot = values.copy()

        if american:
            exercise = intrinsic(S_i)
            early_exercise_nodes += int(np.sum(exercise > values + 1e-12))
            values = np.maximum(values, exercise)

        values = apply_barrier(values, S_i)

    return {
        "price": float(values[0]),
        "S_terminal": S_T,
        "terminal_values": terminal_payoff(S_T),
        "early_exercise_nodes": early_exercise_nodes if american else None,
        "snapshot_values": snapshot,
        "n_nodes": (p.n + 1) * (p.n + 2) // 2,
    }
