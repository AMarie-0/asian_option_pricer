"""
monte_carlo.py — Monte Carlo pricing under risk-neutral GBM.

Why this exists
---------------
The path engine enumerates all 2^n paths exactly, which caps n at ~25
(33M paths, ~800MB). Monte Carlo replaces exhaustive enumeration with
random sampling: cost is O(N_paths * n) instead of O(2^n), so n = 252
(daily monitoring over a year) becomes trivial. The price is no longer
exact — it carries a standard error — but the DISCRETISATION error falls
dramatically, which for Asian options usually dominates.

  n=25  exact enumeration   : zero sampling error, large discretisation error
  n=252 Monte Carlo         : small sampling error, tiny discretisation error

The dynamics simulated are the exact solution of the risk-neutral SDE
    dS = r S dt + sigma S dW
which by Ito's lemma integrates to
    S_t = S_0 exp[(r - sigma^2/2) t + sigma W_t]
so we sample the log-price increments directly. This is exact in
distribution at the monitoring dates — there is NO Euler discretisation
bias in the GBM itself, only in how finely we monitor the average.

Note this uses CONTINUOUS discounting e^{-rT}, matching the GBM used to
generate paths, whereas the binomial tree uses (1+r*dt)^n. The two agree
as n grows but differ slightly at small n.

Variance reduction
------------------
1. ANTITHETIC VARIATES — for every path driven by Z, also run one driven
   by -Z. The two payoffs are negatively correlated, so their average has
   lower variance than two independent draws. Free (same normals reused).

2. CONTROL VARIATE (Asian options) — the GEOMETRIC-average Asian option
   has a closed-form price under GBM, and is very highly correlated with
   the arithmetic one. Pricing both and correcting by the known geometric
   error removes most of the remaining variance. Typically cuts the
   standard error by an order of magnitude.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from src.model.calibration import ModelParams


def estimate_memory_mb(n_paths: int, n_steps: int) -> float:
    """
    Memory footprint of one full path matrix, in MB.

    The array is float64 (8 bytes) of shape (n_paths, n_steps+1), and the
    simulation transiently holds ~3 such arrays (normals, log-path, prices).
    Callers should keep this well under the host's RAM: 500k paths x 252
    steps is ~1 GB per array, which OOMs a 1 GB Streamlit Cloud container.
    """
    return n_paths * (n_steps + 1) * 8 * 3 / 1e6


def simulate_paths(
    p: ModelParams, n_paths: int, n_steps: int | None = None,
    antithetic: bool = True, seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate risk-neutral GBM paths. Returns array (n_paths, n_steps+1)
    including S_0 in column 0.

    With antithetic=True, n_paths is rounded to an even number and half
    the paths are the mirror image (-Z) of the other half.

    Memory: see estimate_memory_mb. For large runs prefer the chunked
    pricers below, which never materialise the full matrix.
    """
    n_steps = n_steps or p.n
    rng = np.random.default_rng(seed)
    dt = p.T / n_steps

    drift = (p.r - 0.5 * p.sigma ** 2) * dt
    vol   = p.sigma * math.sqrt(dt)

    if antithetic:
        half = max(1, n_paths // 2)
        Z = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_incr = drift + vol * Z
    log_path = np.cumsum(log_incr, axis=1)

    S = np.empty((Z.shape[0], n_steps + 1), dtype=np.float64)
    S[:, 0]  = p.S0
    S[:, 1:] = p.S0 * np.exp(log_path)
    return S


def iter_path_chunks(
    p: ModelParams, n_paths: int, n_steps: int,
    antithetic: bool = True, seed: int | None = 42,
    max_chunk_mb: float = 120.0,
):
    """
    Yield path matrices in memory-bounded chunks.

    Large Monte Carlo runs (500k paths x 252 steps ~ 1 GB per array) will
    OOM a small container if materialised at once. This splits the run into
    chunks sized to stay under `max_chunk_mb`, so the caller can accumulate
    payoff statistics without ever holding the full matrix.

    Antithetic pairing is preserved WITHIN each chunk, so the variance
    reduction still applies.
    """
    per_path_mb = (n_steps + 1) * 8 * 3 / 1e6
    chunk = max(2, int(max_chunk_mb / max(per_path_mb, 1e-9)))
    if antithetic and chunk % 2:
        chunk -= 1

    rng = np.random.default_rng(seed)
    dt = p.T / n_steps
    drift = (p.r - 0.5 * p.sigma ** 2) * dt
    vol   = p.sigma * math.sqrt(dt)

    done = 0
    while done < n_paths:
        size = min(chunk, n_paths - done)
        if antithetic:
            half = max(1, size // 2)
            Z = rng.standard_normal((half, n_steps))
            Z = np.concatenate([Z, -Z], axis=0)
        else:
            Z = rng.standard_normal((size, n_steps))

        S = np.empty((Z.shape[0], n_steps + 1), dtype=np.float64)
        S[:, 0]  = p.S0
        S[:, 1:] = p.S0 * np.exp(np.cumsum(drift + vol * Z, axis=1))
        done += Z.shape[0]
        yield S


def _std_error(disc_payoffs: np.ndarray, antithetic: bool) -> float:
    """
    Standard error of the MC estimator.

    IMPORTANT subtlety with antithetic variates: path i and its mirror are
    NEGATIVELY CORRELATED by construction, so they are not independent
    samples. Applying the textbook s/sqrt(N) to the pooled vector assumes
    independence and therefore misstates the error — it hides exactly the
    variance reduction the technique produces.

    The correct treatment is to average each antithetic PAIR into a single
    observation; those pair means ARE independent across pairs, so the usual
    formula applies to them:

        SE = std(pair_means, ddof=1) / sqrt(n_pairs)

    iter_path_chunks lays each chunk out as [Z block | -Z block], so within
    a chunk path k pairs with path k + half.
    """
    n = len(disc_payoffs)
    if not antithetic or n < 4:
        return float(np.std(disc_payoffs, ddof=1) / math.sqrt(n))

    half = n // 2
    pair_means = 0.5 * (disc_payoffs[:half] + disc_payoffs[half:2 * half])
    return float(np.std(pair_means, ddof=1) / math.sqrt(len(pair_means)))


def _summarise(payoffs: np.ndarray, p: ModelParams, extra: dict | None = None,
               antithetic: bool = True) -> dict:
    """Discount, average, and attach a 95% confidence interval."""
    disc = math.exp(-p.r * p.T)
    disc_payoffs = disc * payoffs
    price = float(np.mean(disc_payoffs))
    stderr = _std_error(disc_payoffs, antithetic)
    out = {
        "price": price,
        "std_error": stderr,
        "ci_low":  price - 1.96 * stderr,
        "ci_high": price + 1.96 * stderr,
        "n_paths": len(disc_payoffs),
        "antithetic": antithetic,
        "payoffs": disc_payoffs,
    }
    if extra:
        out.update(extra)
    return out


# ── Asian options ────────────────────────────────────────────────────────────

def geometric_asian_closed_form(p: ModelParams, K: float, n_steps: int) -> float:
    """
    Exact price of a FIXED-STRIKE GEOMETRIC-average Asian call under GBM.

    The geometric average of lognormals is itself lognormal, so a
    Black-Scholes-style formula exists. Used as the control variate for
    the arithmetic Asian option, which has no closed form.
    """
    n = n_steps
    # Effective volatility and drift of the geometric average
    sig_g = p.sigma * math.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_g  = (p.r - 0.5 * p.sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sig_g ** 2

    d1 = (math.log(p.S0 / K) + (mu_g + 0.5 * sig_g ** 2) * p.T) / (sig_g * math.sqrt(p.T))
    d2 = d1 - sig_g * math.sqrt(p.T)
    return math.exp(-p.r * p.T) * (
        p.S0 * math.exp(mu_g * p.T) * norm.cdf(d1) - K * norm.cdf(d2)
    )


def mc_asian_floating(
    p: ModelParams, n_paths: int = 100_000, n_steps: int | None = None,
    antithetic: bool = True, seed: int | None = 42,
) -> dict:
    """Floating-strike Asian call: max(S_T - mean(S), 0)."""
    n_steps = n_steps or p.n
    first, second, term_parts, bar_parts = [], [], [], []
    sample_paths = None
    kept = 0

    for S in iter_path_chunks(p, n_paths, n_steps, antithetic, seed):
        S_bar = S.mean(axis=1)
        pay = np.maximum(S[:, -1] - S_bar, 0.0)
        if antithetic and len(pay) >= 2:
            h = len(pay) // 2
            first.append(pay[:h]); second.append(pay[h:2 * h])
        else:
            first.append(pay)
        if sample_paths is None:
            sample_paths = S[:min(200, len(S))].copy()
        if kept < 100_000:
            term_parts.append(S[:, -1])
            bar_parts.append(S_bar)
            kept += S.shape[0]

    payoffs = np.concatenate(first + second) if second else np.concatenate(first)
    return _summarise(payoffs, p, {
        "n_steps": n_steps,
        "S_terminal": np.concatenate(term_parts),
        "S_bar": np.concatenate(bar_parts),
        "sample_paths": sample_paths,
    }, antithetic)


def mc_asian_fixed(
    p: ModelParams, k_moneyness: float = 1.0, n_paths: int = 100_000,
    n_steps: int | None = None, antithetic: bool = True,
    control_variate: bool = True, seed: int | None = 42,
) -> dict:
    """
    Fixed-strike Asian call: max(mean(S) - K, 0), with optional geometric
    control variate.

    Control variate logic:
        price_corrected = MC_arithmetic - beta * (MC_geometric - exact_geometric)
    with beta = Cov(arith, geo) / Var(geo), the variance-minimising weight.
    """
    n_steps = n_steps or p.n
    K = float(k_moneyness) * p.S0

    a_first, a_second, g_first, g_second, term_parts = [], [], [], [], []
    kept = 0
    for S in iter_path_chunks(p, n_paths, n_steps, antithetic, seed):
        a = np.maximum(S.mean(axis=1) - K, 0.0)
        g = None
        if control_variate:
            # Geometric average excludes S_0 to match the closed form
            geo_avg = np.exp(np.log(S[:, 1:]).mean(axis=1))
            g = np.maximum(geo_avg - K, 0.0)
        if antithetic and len(a) >= 2:
            h = len(a) // 2
            a_first.append(a[:h]); a_second.append(a[h:2 * h])
            if g is not None:
                g_first.append(g[:h]); g_second.append(g[h:2 * h])
        else:
            a_first.append(a)
            if g is not None:
                g_first.append(g)
        if kept < 100_000:
            term_parts.append(S[:, -1])
            kept += S.shape[0]

    arith = np.concatenate(a_first + a_second) if a_second else np.concatenate(a_first)
    geo_parts = ([np.concatenate(g_first + g_second)] if g_second
                 else ([np.concatenate(g_first)] if g_first else []))
    result = _summarise(arith, p, {"n_steps": n_steps, "strike": K,
                                   "S_terminal": np.concatenate(term_parts)},
                        antithetic)
    result["price_naive"] = result["price"]
    result["std_error_naive"] = result["std_error"]

    if control_variate and geo_parts:
        geo  = np.concatenate(geo_parts)
        disc = math.exp(-p.r * p.T)
        a_d, g_d = disc * arith, disc * geo

        var_g = np.var(g_d, ddof=1)
        if var_g > 1e-12:
            beta = np.cov(a_d, g_d, ddof=1)[0, 1] / var_g
            exact_geo = geometric_asian_closed_form(p, K, n_steps)
            corrected = a_d - beta * (g_d - exact_geo)
            price  = float(np.mean(corrected))
            stderr = _std_error(corrected, antithetic)
            result.update({
                "price": price, "std_error": stderr,
                "ci_low": price - 1.96 * stderr,
                "ci_high": price + 1.96 * stderr,
                "control_variate": True,
                "beta": float(beta),
                "geometric_exact": exact_geo,
                "variance_reduction": (result["std_error_naive"] / stderr
                                       if stderr > 0 else float("nan")),
            })
    return result


# ── Vanilla & other options ──────────────────────────────────────────────────

def _chunked_payoffs(p, n_paths, n_steps, antithetic, seed, payoff_of_chunk):
    """
    Run a payoff function over memory-bounded chunks and concatenate the
    resulting payoff vectors (payoffs are 1-D, so this stays small even
    when the path matrices would not).

    Also keeps terminal prices for plotting, capped so the UI never has to
    hold a huge array.
    """
    # Each chunk is laid out [Z block | -Z block]. To keep global pair
    # adjacency (path k pairs with path k+half) we accumulate the first
    # halves and second halves separately, then concatenate.
    first, second, terminal_parts = [], [], []
    kept = 0
    for S in iter_path_chunks(p, n_paths, n_steps, antithetic, seed):
        pay = payoff_of_chunk(S)
        if antithetic and len(pay) >= 2:
            h = len(pay) // 2
            first.append(pay[:h]); second.append(pay[h:2 * h])
        else:
            first.append(pay)
        if kept < 100_000:
            terminal_parts.append(S[:, -1])
            kept += S.shape[0]
    payoffs = np.concatenate(first + second) if second else np.concatenate(first)
    return (payoffs,
            np.concatenate(terminal_parts) if terminal_parts else np.array([]))


def mc_european(
    p: ModelParams, k_moneyness: float = 1.0, kind: str = "call",
    n_paths: int = 100_000, n_steps: int | None = None,
    antithetic: bool = True, seed: int | None = 42,
) -> dict:
    """European call/put — MC is overkill here but validates the engine."""
    n_steps = n_steps or p.n
    K = float(k_moneyness) * p.S0
    fn = (lambda S: np.maximum(S[:, -1] - K, 0.0)) if kind == "call" \
        else (lambda S: np.maximum(K - S[:, -1], 0.0))
    payoffs, ST = _chunked_payoffs(p, n_paths, n_steps, antithetic, seed, fn)
    return _summarise(payoffs, p, {"n_steps": n_steps, "strike": K,
                                   "S_terminal": ST}, antithetic)


def mc_lookback_floating(
    p: ModelParams, n_paths: int = 100_000, n_steps: int | None = None,
    antithetic: bool = True, seed: int | None = 42,
) -> dict:
    """Lookback call: S_T - min(S). Discrete monitoring at n_steps dates."""
    n_steps = n_steps or p.n
    payoffs, ST = _chunked_payoffs(
        p, n_paths, n_steps, antithetic, seed,
        lambda S: S[:, -1] - S.min(axis=1))
    return _summarise(payoffs, p, {"n_steps": n_steps, "S_terminal": ST}, antithetic)


def mc_barrier_up_out(
    p: ModelParams, k_moneyness: float = 1.0, h_moneyness: float = 1.3,
    rebate: float = 0.0, n_paths: int = 100_000, n_steps: int | None = None,
    antithetic: bool = True, seed: int | None = 42,
) -> dict:
    """Up-and-out barrier call, monitored discretely at n_steps dates."""
    n_steps = n_steps or p.n
    K = float(k_moneyness) * p.S0
    H = float(h_moneyness) * p.S0
    knock_counts = {"knocked": 0, "total": 0}

    def fn(S):
        knocked = (S.max(axis=1) >= H)
        knock_counts["knocked"] += int(knocked.sum())
        knock_counts["total"]   += len(knocked)
        return np.where(knocked, rebate, np.maximum(S[:, -1] - K, 0.0))

    payoffs, ST = _chunked_payoffs(p, n_paths, n_steps, antithetic, seed, fn)
    return _summarise(payoffs, p, {
        "n_steps": n_steps, "strike": K, "barrier": H,
        "knockout_rate": knock_counts["knocked"] / max(knock_counts["total"], 1),
        "S_terminal": ST,
    }, antithetic)


# Registry-key -> MC pricer, so the UI can offer MC wherever it applies
MC_PRICERS = {
    "asian_float":     lambda p, **kw: mc_asian_floating(p, **kw),
    "asian_fixed":     lambda p, k_moneyness=1.0, **kw: mc_asian_fixed(p, k_moneyness, **kw),
    "lookback_float":  lambda p, **kw: mc_lookback_floating(p, **kw),
    "euro_call":       lambda p, k_moneyness=1.0, **kw: mc_european(p, k_moneyness, "call", **kw),
    "euro_put":        lambda p, k_moneyness=1.0, **kw: mc_european(p, k_moneyness, "put", **kw),
    "barrier_uo_call": lambda p, k_moneyness=1.0, h_moneyness=1.3, rebate=0.0, **kw:
                       mc_barrier_up_out(p, k_moneyness, h_moneyness, rebate, **kw),
}


def convergence_study(
    p: ModelParams, mc_fn, path_counts=(1_000, 5_000, 10_000, 50_000, 100_000),
    n_steps: int | None = None, **kwargs,
) -> dict:
    """Price at increasing N to show the 1/sqrt(N) error decay."""
    rows = []
    for N in path_counts:
        res = mc_fn(p, n_paths=N, n_steps=n_steps, **kwargs)
        rows.append({
            "n_paths": N, "price": res["price"],
            "std_error": res["std_error"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
        })
    return {"rows": rows}
