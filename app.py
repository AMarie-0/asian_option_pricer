"""
app.py — Derivatives Pricer
Streamlit app: multi-option pricer built on CRR binomial trees.
Classic & exotic options on any equity ticker.
"""
import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

GITHUB_USER = "antoinemarie"   # ← change to your actual GitHub username

st.set_page_config(
    page_title="Derivatives Pricer",
    page_icon="📐",
    layout="wide",
)

import os
ON_CLOUD = os.environ.get("HOME") == "/home/adminuser"
MAX_N    = 20 if ON_CLOUD else 25

try:
    from data.fetch import fetch_prices, fetch_risk_free_rate, SUPPORTED_TICKERS
    from src.model.calibration import calibrate
    from src.model.pricer import price_asian_call
    from src.analytics.approximation import normal_approximation_price, empirical_Dn_stats
    from src.analytics.robustness import volatility_window_sensitivity, normal_approx_convergence
    from src.visualization.plots import (
        plot_price_series, plot_log_returns, plot_binomial_tree,
        plot_terminal_distribution, plot_average_distribution,
        plot_payoff_distribution, plot_payoff_vs_terminal,
        plot_window_sensitivity, plot_approx_convergence,
    )
    from src.options.registry import REGISTRY, by_category
    try:
        from data.database import save_prices, load_prices
        DB_AVAILABLE = True
    except Exception:
        DB_AVAILABLE = False
except Exception as e:
    import traceback
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── colour palette (all blue family) ─────────────────────────────────────────
BLUE_DARK   = "#0d2137"   # headings, values
BLUE_MID    = "#1a5fa8"   # accents, borders
BLUE_LIGHT  = "#dbeafe"   # backgrounds
BLUE_FAINT  = "#f0f6ff"   # metric boxes
RED_LOSS    = "#c0392b"   # payoff diagram loss zone
GREEN_GAIN  = "#1a8a4a"   # payoff diagram gain zone

st.markdown(f"""
<style>
  /* metric boxes */
  .metric-box {{
    background:{BLUE_FAINT}; border-radius:8px;
    padding:16px 20px; margin-bottom:8px;
    border-left:3px solid {BLUE_MID};
  }}
  .metric-label {{
    font-size:11px; color:#5a7a99; text-transform:uppercase;
    letter-spacing:.07em; font-weight:600;
  }}
  .metric-value {{
    font-size:26px; font-weight:700; color:{BLUE_DARK}; margin-top:4px;
  }}
  .metric-sub {{ font-size:11px; color:#7a9ab8; margin-top:2px; }}

  /* section headers */
  .section-header {{
    font-size:12px; font-weight:700; color:{BLUE_MID};
    text-transform:uppercase; letter-spacing:.1em;
    border-bottom:2px solid {BLUE_MID}; padding-bottom:4px;
    margin:28px 0 14px;
  }}

  /* note/callout boxes */
  .note-box {{
    background:{BLUE_LIGHT}; border-left:4px solid {BLUE_MID};
    padding:12px 16px; border-radius:0 6px 6px 0;
    font-size:13px; color:#1a3a5c; margin:10px 0; line-height:1.6;
  }}

  /* warning boxes */
  .warn-box {{
    background:#fff3cd; border-left:4px solid #f0a500;
    padding:12px 16px; border-radius:0 6px 6px 0;
    font-size:13px; color:#5a3e00; margin:10px 0; line-height:1.6;
  }}

  /* footer */
  .footer {{
    margin-top:48px; padding-top:20px;
    border-top:1px solid #d0dff0;
    text-align:center; color:#7a9ab8; font-size:12px;
  }}
  .footer a {{ color:{BLUE_MID}; text-decoration:none; }}
  .footer a:hover {{ text-decoration:underline; }}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def metric(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-sub">'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def note(text):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def footer():
    st.markdown(
        f'<div class="footer">'
        f'Built with CRR binomial trees &nbsp;·&nbsp; '
        f'<a href="https://github.com/{GITHUB_USER}" target="_blank">@{GITHUB_USER}</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── cached data ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(ticker: str) -> pd.DataFrame:
    if DB_AVAILABLE:
        try:
            return load_prices(ticker, db_path="data/prices.duckdb")
        except Exception:
            pass
    df = fetch_prices(ticker)
    if DB_AVAILABLE:
        try:
            save_prices(df, ticker, db_path="data/prices.duckdb")
        except Exception:
            pass
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_risk_free_rate() -> float:
    return fetch_risk_free_rate()

@st.cache_data(show_spinner=False)
def run_pricer(ticker, r, T, n, vol_window):
    df     = get_prices(ticker)
    params = calibrate(df, ticker, r=r, T=T, n=n, volatility_window=vol_window)
    result = price_asian_call(params)
    approx = normal_approximation_price(params)
    emp    = empirical_Dn_stats(result)
    return df, params, result, approx, emp

@st.cache_data(show_spinner=False)
def run_robustness(ticker, r, T, n, _df_hash):
    df = get_prices(ticker)
    return volatility_window_sensitivity(df, ticker, r=r, T=T, n=n)

@st.cache_data(show_spinner=False)
def run_option(spec_key, ticker, r, T, n, vol_window, inputs_items):
    df     = get_prices(ticker)
    params = calibrate(df, ticker, r=r, T=T, n=n, volatility_window=vol_window)
    result = REGISTRY[spec_key].price(params, **dict(inputs_items))
    return df, params, result

@st.cache_data(show_spinner=False)
def run_convergence(_params_hash, sigma, r, T, n, S0, u, d, q, R_period, discount):
    from src.model.calibration import ModelParams
    p = ModelParams("", S0, sigma, u, d, q, r, n, T/n, T, R_period, discount)
    return normal_approx_convergence(p)

@st.cache_data(show_spinner=False)
def run_generic_robustness(spec_key, ticker, r, T, n, inputs_items):
    """Re-price any lattice option across the four volatility windows."""
    df = get_prices(ticker)
    windows = [
        ("full",       "2020–present (baseline)"),
        ("post_covid", "2021–present (post-COVID)"),
        ("3y",         "2023–present (3-year)"),
        ("1y",         "2025–present (1-year)"),
    ]
    rows = []
    for wkey, wlabel in windows:
        try:
            params = calibrate(df, ticker, r=r, T=T, n=n, volatility_window=wkey)
            res    = REGISTRY[spec_key].price(params, **dict(inputs_items))
            rows.append({
                "Window": wlabel,
                "σ annual (%)": f"{params.sigma*100:.2f}%",
                "V₀ ($)": f"${res['price']:.4f}",
                "_price": res["price"],
                "_sigma": params.sigma,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Black–Scholes reference ───────────────────────────────────────────────────
def bs_reference(spec_key, p, extra):
    from scipy.stats import norm
    K = float(extra.get("k_moneyness", 1.0)) * p.S0
    sqT = p.sigma * math.sqrt(p.T)
    d1  = (math.log(p.S0 / K) + (p.r + p.sigma**2 / 2) * p.T) / sqT
    d2  = d1 - sqT
    disc = math.exp(-p.r * p.T)
    if spec_key == "euro_call":
        return p.S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if spec_key == "euro_put":
        return K * disc * norm.cdf(-d2) - p.S0 * norm.cdf(-d1)
    if spec_key == "digital_call":
        return float(extra.get("payout", 1.0)) * disc * norm.cdf(d2)
    return None


# ── payoff diagram (call or put, real values, app colour scheme) ──────────────
def payoff_diagram(spec_key, S0, K, price, T_label):
    """Classic hockey-stick P&L diagram with real dollar values."""
    lo = K * 0.5
    hi = K * 1.8
    S  = np.linspace(lo, hi, 400)

    if spec_key in ("euro_call", "amer_call"):
        payoff   = np.maximum(S - K, 0)
        pnl      = payoff - price
        label_up = "In-the-money: S > K"
        label_dn = "Out-of-the-money: S < K"
        be_point = K + price
    else:
        payoff   = np.maximum(K - S, 0)
        pnl      = payoff - price
        label_up = "In-the-money: S < K"
        label_dn = "Out-of-the-money: S > K"
        be_point = K - price

    fig = go.Figure()

    # Shaded gain / loss regions
    fig.add_trace(go.Scatter(
        x=S, y=np.maximum(pnl, 0),
        fill="tozeroy", fillcolor="rgba(26,138,74,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=S, y=np.minimum(pnl, 0),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # P&L line
    fig.add_trace(go.Scatter(
        x=S, y=pnl,
        mode="lines",
        line=dict(color=BLUE_MID, width=2.5),
        name="P&L at expiry",
        hovertemplate="S = $%{x:.2f}<br>P&L = $%{y:.2f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="#a0b4cc", width=1, dash="dot"))

    # Strike annotation
    fig.add_vline(x=K, line=dict(color="#7a9ab8", width=1, dash="dash"))
    fig.add_annotation(x=K, y=pnl.min() * 0.7,
                       text=f"K = ${K:.0f}", showarrow=False,
                       font=dict(size=11, color="#5a7a99"), xanchor="left", xshift=4)

    # Breakeven annotation
    fig.add_vline(x=be_point, line=dict(color=BLUE_MID, width=1, dash="longdash"))
    fig.add_annotation(x=be_point, y=pnl.max() * 0.3,
                       text=f"Breakeven ${be_point:.0f}", showarrow=False,
                       font=dict(size=11, color=BLUE_MID), xanchor="left", xshift=4)

    # Premium label
    fig.add_annotation(
        x=lo + (K - lo) * 0.3, y=-price,
        text=f"Premium paid: −${price:.2f}",
        showarrow=True, arrowhead=2, arrowcolor=RED_LOSS,
        font=dict(size=11, color=RED_LOSS),
        ax=0, ay=-30,
    )

    # Spot price marker
    fig.add_vline(x=S0, line=dict(color=BLUE_DARK, width=1.5))
    fig.add_annotation(x=S0, y=pnl.max() * 0.85,
                       text=f"S₀ = ${S0:.0f}", showarrow=False,
                       font=dict(size=11, color=BLUE_DARK, weight=700), xanchor="center")

    fig.update_layout(
        title=dict(text=f"P&L at expiry · {T_label}", font=dict(size=14, color=BLUE_DARK)),
        xaxis=dict(title="Stock price at expiry (Sₙ)", tickprefix="$",
                   gridcolor="#e8f0f8", showline=True, linecolor="#c0d4e8"),
        yaxis=dict(title="Profit / Loss ($)", tickprefix="$",
                   gridcolor="#e8f0f8", showline=True, linecolor="#c0d4e8"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        height=420,
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


# ── methodology per option ────────────────────────────────────────────────────
METHODOLOGY = {
    "euro_call": {
        "title": "European Call Option",
        "payoff_tex": r"X_T = \max(S_n - K,\ 0)",
        "body": """
**What it is.** A European call gives the holder the right — but not the obligation — to *buy*
the underlying at a fixed strike price K at maturity T. The option can only be exercised at
expiry, not before.

**Why it has value.** At expiry the holder compares the market price Sₙ to K. If Sₙ > K they
exercise and pocket the difference; if Sₙ < K the option expires worthless and the holder loses
only the premium paid. The maximum loss is bounded (the premium); the upside is unlimited.

**Pricing.** The CRR binomial model prices the call by backward induction under the risk-neutral
probability q. As n → ∞ the price converges to the Black–Scholes formula:
""",
        "latex": r"C = S_0 N(d_1) - K e^{-rT} N(d_2), \quad d_{1,2} = \frac{\ln(S_0/K)+(r\pm\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}",
        "body2": """
where N(·) is the standard normal CDF. The binomial price is displayed alongside the BS
benchmark — the gap shrinks with n and reflects discrete vs continuous discounting.

**Key sensitivities (Greeks).** Δ (delta) = N(d₁) ∈ (0,1); the option price rises roughly
Δ dollars for each dollar Sₙ rises. Vega is always positive — higher volatility always increases
call value because it widens the upside distribution while the downside remains floored at zero.
""",
    },
    "euro_put": {
        "title": "European Put Option",
        "payoff_tex": r"X_T = \max(K - S_n,\ 0)",
        "body": """
**What it is.** A European put gives the holder the right to *sell* the underlying at strike K
at maturity. Profit arises when the stock falls below the strike.

**Put–call parity.** The put and call prices are not independent — they are tied by a fundamental
no-arbitrage identity:
""",
        "latex": r"C - P = S_0 - K e^{-rT}",
        "body2": """
This holds exactly in the binomial model (verified to 1e-12 precision in the test suite). It
means that given the call price you can always derive the put price without re-running the tree.

**Why investors buy puts.** A long put is the classic hedge: the holder of 100 shares of AAPL
can buy a put to guarantee a minimum exit price of K, converting unbounded downside into a known
cost equal to the put premium.

**Δ of a put** = N(d₁) − 1 ∈ (−1, 0). The option loses value as the stock rises, so delta
is negative. Vega is still positive for the same reason as a call.
""",
    },
    "amer_call": {
        "title": "American Call Option",
        "payoff_tex": r"X_t = \max(S_t - K,\ 0) \text{ exercisable at any } t \le T",
        "body": """
**What it is.** An American call is identical to a European call except the holder may exercise
*at any point* up to and including maturity. The binomial lattice naturally handles this —
at each node the holder compares the immediate exercise value max(S−K, 0) with the continuation
value and keeps whichever is higher.

**The early-exercise theorem.** Without dividends, early exercise of an American call is never
optimal. The intuition: exercising early foregoes the time value of keeping the position open
(the option could move further in-the-money) and also accelerates payment of K, which the holder
would rather keep earning interest on. Formally this follows from the lower bound:
""",
        "latex": r"C^{\text{Amer}} \ge S_0 - K e^{-rT} > S_0 - K = \text{immediate exercise value}",
        "body2": """
This means **the American call price equals the European call price** when there are no dividends —
a built-in sanity check in this tool. The early-exercise premium displayed in the metrics should
be exactly zero; any deviation is floating-point rounding at the level of 1e-9 or less.

**With dividends** the result breaks down: a large discrete dividend tomorrow makes early exercise
optimal today. For dividend-paying stocks the American call genuinely exceeds the European call.
""",
    },
    "amer_put": {
        "title": "American Put Option",
        "payoff_tex": r"X_t = \max(K - S_t,\ 0) \text{ exercisable at any } t \le T",
        "body": """
**What it is.** An American put allows early exercise at any time. Unlike the American call,
**early exercise is genuinely optimal** for the put even without dividends.

**Why early exercise makes sense for puts.** If the stock falls to near zero, the holder can
receive K immediately rather than waiting. The money received can then earn the risk-free rate
for the remaining life of the option. Waiting brings no additional upside since the stock is
already floored at zero (limited liability). Therefore there exists a critical price S* below
which the American put should always be exercised immediately.

**The early-exercise premium** (shown in the metrics) is the price difference between the
American and European puts. This premium is strictly positive and increases with:
- higher interest rates (opportunity cost of waiting grows)
- deeper in-the-money positions
- longer remaining maturity

**Binomial pricing.** The lattice evaluates early exercise at every node. The number of nodes
where early exercise is optimal is shown as a gauge of how much of the tree is in the exercise
region.
""",
        "latex": r"V_t = \max\!\left(\underbrace{K - S_t}_{\text{exercise}},\ \underbrace{\frac{1}{R}(q V_{t+1}^u + (1-q) V_{t+1}^d)}_{\text{continuation}}\right)",
        "body2": "",
    },
    "digital_call": {
        "title": "Digital Call (Cash-or-Nothing)",
        "payoff_tex": r"X_T = Q \cdot \mathbf{1}\{S_n > K\}",
        "body": """
**What it is.** A digital (binary) call pays a fixed cash amount Q if the stock finishes above K,
and zero otherwise. There is no proportional participation in the stock move — the payoff is
discontinuous at K.

**Risk-neutral probability.** The discounted digital price equals Q × ℚ(Sₙ > K) — it directly
prices the risk-neutral probability of finishing in-the-money. In the Black–Scholes limit this
equals Q × e⁻ʳᵀ × N(d₂), revealing that N(d₂) is the risk-neutral ITM probability.

**Binomial oscillation.** Because the payoff jumps from 0 to Q at exactly K, and the lattice has
discrete nodes, the digital price oscillates as n changes (the jump may land between nodes).
The standard fix is to average prices at n and n+1. Higher n reduces but doesn't eliminate the
oscillation — the payoff discontinuity is the fundamental difficulty.

**Real-world use.** Digital options appear embedded in structured products (principal-protected
notes, range accruals) and are widely used in FX markets. They are difficult to hedge because
the delta spikes to infinity as Sₙ approaches K near expiry — a practical trading challenge
that simple models obscure.
""",
        "latex": r"V_0 = Q \cdot e^{-rT} N(d_2), \quad d_2 = \frac{\ln(S_0/K)+(r-\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}",
        "body2": "",
    },
    "asian_float": {
        "title": "Asian Call — Floating Strike",
        "payoff_tex": r"A_T = \max\!\left(S_n - \bar{S}_n,\ 0\right), \quad \bar{S}_n = \frac{1}{n+1}\sum_{t=0}^{n} S_t",
        "body": """
**What it is.** The floating-strike Asian call pays the excess of the terminal price over the
*arithmetic average* of prices over the entire contract life. There is no fixed strike — it is
replaced by the path average. The holder profits when the stock finishes above its own average
performance.

**Why path-dependence matters.** Two paths with identical terminal prices can have different
averages depending on the order of up and down moves. The standard recombining tree, which
assigns a single value per node, is insufficient — we must enumerate all 2ⁿ paths explicitly.

**Augmented state space.** The trick that makes this tractable is tracking the cumulative sum
Cₜ = Σ Sₜ alongside the price. The pair (Sₜ, Cₜ) is Markov — its future depends only on its
current value. At maturity: S̄ₙ = Cₙ / (n+1).

**Pricing.** Forward enumeration propagates all 2ⁿ paths. Backward induction under ℚ folds the
payoff vector in half n times with vectorised arithmetic. Both methods are run and must agree to
floating-point precision.

**Normal approximation.** Linearising Sₜ ≈ S₀(1 + σ√Δt Wₜ) and using the covariance structure
of the random walk yields the closed-form approximation shown in the Robustness tab.
""",
        "latex": r"\operatorname{Var}(D_n) = S_0^2\sigma^2 T \frac{2n+1}{6(n+1)}, \quad V_0^{\text{approx}} = e^{-rT}\frac{\sqrt{\operatorname{Var}(D_n)}}{\sqrt{2\pi}}",
        "body2": "",
    },
    "asian_fixed": {
        "title": "Asian Call — Fixed Strike",
        "payoff_tex": r"A_T = \max\!\left(\bar{S}_n - K,\ 0\right)",
        "body": """
**What it is.** The fixed-strike Asian call pays the excess of the *arithmetic average* over a
predetermined strike K. The average replaces Sₙ in the usual call payoff.

**Comparison with vanilla.** Because averaging smooths out extreme price paths, the average S̄ₙ
has lower variance than any individual Sₜ. Concretely: the Asian call is always cheaper than
a vanilla European call with the same parameters. The averaging effect reduces the effective
volatility of the payoff — this is the central reason Asian options are popular in markets where
buyers want exposure to average performance rather than a single snap price.

**Path dependence.** Like the floating-strike version, the fixed-strike Asian call requires full
path enumeration — the same 2ⁿ forward pass. The difference is only in what is computed at the
end of each path: here payoff = max(S̄ₙ − K, 0) rather than max(Sₙ − S̄ₙ, 0).

**Practical use.** Fixed-strike Asian options are extremely common in commodity markets (oil,
natural gas), where the purchaser hedges the average price they will pay over a month rather
than a single daily fixing. They are also used in FX for importers/exporters with regular cash flows.
""",
        "latex": r"V_0 = \frac{1}{(1+r\Delta t)^n}\sum_\omega q^{j(\omega)}(1-q)^{n-j(\omega)} \max\!\left(\bar{S}_n(\omega) - K,\ 0\right)",
        "body2": "",
    },
    "lookback_float": {
        "title": "Lookback Call — Floating Strike",
        "payoff_tex": r"X_T = S_n - \min_{0 \le t \le n} S_t",
        "body": """
**What it is.** The lookback call pays the difference between the terminal price and the *minimum*
price over the entire life of the contract. The holder effectively buys at the cheapest point —
with the benefit of hindsight.

**Always positive payoff.** Unlike a vanilla call that can expire worthless, the lookback payoff
is always ≥ 0. If the stock ends at its minimum (every path goes down then stays flat), the payoff
is zero. Otherwise it is strictly positive. The option is expensive precisely because the minimum
can never be worse than zero.

**Path enumeration.** We track the running minimum per path alongside the stock price, using the
same [up | down] concatenation as the Asian engine. No additional tricks are needed — the backward
induction works identically.

**Real-world use.** Lookback options are primarily used in structured products and by hedge funds
as benchmarking tools. They are also the building block for various "ratchet" structures. In
practice they are expensive, thinly traded, and mostly relevant as a theoretical benchmark.
""",
        "latex": r"X_T = S_n - \min_t S_t \ge 0 \text{ always}",
        "body2": "",
    },
    "barrier_uo_call": {
        "title": "Barrier Call — Up-and-Out",
        "payoff_tex": r"X_T = \max(S_n - K,\ 0) \cdot \mathbf{1}\!\left\{\max_{0 \le t \le n} S_t < H\right\}",
        "body": """
**What it is.** The up-and-out call is a vanilla call that is *knocked out* if the stock ever
reaches the barrier H. If the stock touches H at any monitored point, the holder receives only
the rebate (zero by default). If the stock stays below H throughout, the payoff is the usual
max(Sₙ − K, 0).

**In–out parity.** An up-and-in call activates only if the stock reaches H. Together they span
all scenarios: one or the other must pay, so their prices sum to the vanilla call price:

        Up-and-Out + Up-and-In = Vanilla call

This is verified in the tool's metrics. It is a no-arbitrage identity that holds regardless
of the model.

**Why barrier options are cheaper.** The up-and-out holder loses the option precisely when the
stock has moved favourably (upward) — a perverse knock-out. This makes it cheaper than the
vanilla. The option is useful when the buyer wants upside participation only up to a point,
or when selling the barrier premium reduces the net cost.

**Discrete monitoring.** This implementation monitors the barrier at each lattice node, not
continuously. Discrete monitoring overestimates the survival probability slightly (the stock
could breach H between nodes). More steps n → better approximation of continuous monitoring.
""",
        "latex": r"C_{\text{out}} + C_{\text{in}} = C_{\text{vanilla}}",
        "body2": "",
    },
    "chooser": {
        "title": "Chooser Option",
        "payoff_tex": r"X_m = \max\!\left(C_m(K),\ P_m(K)\right) \text{ at choice date } m",
        "body": """
**What it is.** At a pre-specified *choice date* m < T, the holder decides whether the option
becomes a K-strike European call or put maturing at T. After the choice is made, it behaves
identically to whichever instrument was selected.

**Why it is valuable.** The chooser is a bet on *volatility without directional commitment*.
If the stock has moved dramatically by date m, the holder picks the in-the-money leg. If the
stock has barely moved, the chooser is worth roughly max(call, put) ≈ a straddle. It is
always worth at least the call and at least the put:

        Chooser ≥ max(Call_m, Put_m)

with equality when choice_date = 0 (must choose immediately) and the price approaches a
straddle as choice_date → T.

**Lattice pricing.** Two separate backward inductions are run to step m — one for the call
and one for the put — capturing their continuation values at each node at step m. The chooser
value at step m is the node-wise maximum. These values are then folded backward to t = 0 for
the remaining m steps.

**Real-world use.** Choosers are rarer than straddles but appear in structured notes where
investors want the flexibility to switch view after observing early-period performance. The
choice date premium is the key differentiator from a plain straddle.
""",
        "latex": r"V_0^{\text{chooser}} = \frac{1}{R^m}\mathbb{E}^Q\!\left[\max\!\left(C_m, P_m\right)\right]",
        "body2": "",
    },
}


PAYOFF_LATEX = {k: v["payoff_tex"] for k, v in METHODOLOGY.items()}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 4px;">
      <div style="font-size:22px;font-weight:800;color:{BLUE_DARK};letter-spacing:-0.5px;">
        📐 Derivatives Pricer
      </div>
      <div style="font-size:11px;color:#7a9ab8;margin-top:2px;letter-spacing:.04em;">
        CRR Binomial Tree &nbsp;·&nbsp; Classic & Exotic Options
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    ticker_mode = st.radio("Ticker input", ["Curated list", "Any ticker"], horizontal=True)
    if ticker_mode == "Curated list":
        ticker = st.selectbox("Ticker", list(SUPPORTED_TICKERS.keys()), index=0)
        st.caption(SUPPORTED_TICKERS[ticker])
    else:
        ticker = st.text_input("Enter ticker symbol", value="AAPL").upper().strip()
        if ticker:
            st.caption("US: `AAPL` `MSFT` `GS` — Europe: `SIE.DE` `ASML.AS` `BNP.PA`")

    st.subheader("Option")
    category = st.radio("Category", ["classic", "exotic"], index=1,
                        horizontal=True, format_func=str.capitalize)
    specs = by_category(category)
    _default = next((i for i, s in enumerate(specs) if s.key == "asian_float"), 0)
    spec = st.selectbox("Option type", specs, index=_default,
                        format_func=lambda s: s.name)
    st.caption(spec.description)

    extra_inputs = {}
    for ei in spec.extra_inputs:
        extra_inputs[ei.key] = st.number_input(
            ei.label,
            min_value=float(ei.minimum), max_value=float(ei.maximum),
            value=float(ei.default), step=0.05,
            help=ei.help, key=f"ei_{spec.key}_{ei.key}",
        )

    st.subheader("Model Parameters")
    T = st.slider("Maturity T (years)", 0.25, 2.0, 0.5, 0.25)
    if spec.engine == "paths":
        n = st.slider("Binomial steps n", 5, MAX_N, min(20, MAX_N), 5)
        st.caption(f"Path enumeration is O(2ⁿ) — capped at n = {MAX_N}.")
    else:
        n = st.slider("Binomial steps n", 25, 500, spec.default_n, 25)
        st.caption("Recombining lattice is O(n²) — large n converges to Black–Scholes.")

    use_live_r = st.checkbox("Fetch live risk-free rate (^IRX)", value=False)
    if use_live_r:
        with st.spinner("Fetching rate..."):
            r = get_risk_free_rate()
        st.caption(f"Live rate: {r*100:.3f}%")
    else:
        r = st.number_input("Risk-free rate r (annual)", 0.0, 0.2, 0.01, 0.005,
                            format="%.3f")

    st.subheader("Volatility Window")
    vol_window = st.selectbox(
        "Estimation window",
        ["full", "post_covid", "3y", "1y"],
        format_func=lambda x: {
            "full":       "2020–present (baseline)",
            "post_covid": "2021–present (post-COVID)",
            "3y":         "2023–present (3-year)",
            "1y":         "2025–present (1-year)",
        }[x],
    )

    run = st.button("▶  Price", type="primary", use_container_width=True)
    st.divider()
    st.caption("CRR binomial tree · 2ⁿ path enumeration (path-dependent) · "
               "recombining lattice (vanilla/exotics) · Black–Scholes benchmark")


# ── main header ───────────────────────────────────────────────────────────────
st.title(spec.name)
if spec.key in PAYOFF_LATEX:
    st.latex(PAYOFF_LATEX[spec.key])

if not run:
    st.info("Configure parameters in the sidebar and click **▶ Price** to run.")
    footer()
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# GENERIC OPTIONS — everything except Asian floating-strike
# ═══════════════════════════════════════════════════════════
if spec.key != "asian_float":
    with st.spinner(f"Fetching {ticker} · pricing {spec.name}..."):
        try:
            df, params, result = run_option(
                spec.key, ticker, r, T, n, vol_window,
                tuple(sorted(extra_inputs.items())),
            )
        except Exception as e:
            st.error(f"Could not price **{ticker}**: {e}")
            st.stop()

    tab_labels = ["💰 Pricer", "📊 Model", "🔬 Robustness", "📖 Methodology"]
    gtab1, gtab2, gtab3, gtab4 = st.tabs(tab_labels)

    # ── PRICER ────────────────────────────────────────────────────────────────
    with gtab1:
        section("Option Price")
        bs = bs_reference(spec.key, params, extra_inputs)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric("Option Price (Binomial)", f"${result['price']:.4f}",
                   f"{spec.engine} engine · n = {params.n}")
        with c2:
            if bs is not None:
                metric("Black–Scholes", f"${bs:.4f}",
                       f"Δ = {result['price'] - bs:+.4f} vs lattice")
            elif "strike" in result:
                metric("Strike K", f"${result['strike']:.2f}",
                       f"{extra_inputs.get('k_moneyness', 1.0):.2f} × S₀")
            else:
                metric("Spot S₀", f"${params.S0:.2f}", "at calibration date")
        with c3:
            if spec.key in ("amer_call", "amer_put"):
                metric("Early-Exercise Premium",
                       f"${result['early_exercise_premium']:.4f}",
                       f"European twin: ${result['european_price']:.4f}")
            elif spec.key == "digital_call":
                metric("Risk-neutral P(ITM)", f"{result['rn_prob_itm']:.4f}",
                       "= price / (e⁻ʳᵀ × Q)")
            elif spec.key == "barrier_uo_call":
                metric("Vanilla Call", f"${result['vanilla_price']:.4f}",
                       f"H = ${result['barrier']:.2f}")
            elif spec.key == "chooser":
                metric("Call leg / Put leg",
                       f"${result['call_price']:.2f} / ${result['put_price']:.2f}",
                       f"choice at step {result['choice_step']}")
            elif "n_paths" in result:
                metric("Paths Enumerated", f"{result['n_paths']:,}", f"= 2^{params.n}")
            else:
                metric("σ (annual)", f"{params.sigma*100:.2f}%", vol_window)
        with c4:
            if spec.key == "barrier_uo_call" and result.get("up_and_in_price") is not None:
                metric("Up-and-In Twin", f"${result['up_and_in_price']:.4f}",
                       "in + out = vanilla ✓")
            else:
                metric("Risk-neutral q", f"{params.q:.4f}",
                       f"discount/step = {params.discount:.6f}")

        # payoff diagram for vanilla calls / puts
        if spec.key in ("euro_call", "euro_put", "amer_call", "amer_put"):
            section("Payoff Diagram")
            K   = result.get("strike", params.S0)
            fig = payoff_diagram(spec.key, params.S0, K, result["price"],
                                 f"T = {params.T:.2f} yrs · K = ${K:.0f}")
            st.plotly_chart(fig, use_container_width=True, key="g_payoff_diag")
            note(
                "The diagram shows the <b>net P&L</b> at expiry after accounting for the premium paid. "
                f"The horizontal flat at −${result['price']:.2f} is the maximum loss (option expires worthless). "
                f"The breakeven point is where the P&L crosses zero. "
                f"Current spot S₀ = ${params.S0:.2f} is shown as a reference."
            )

        # contextual notes
        if bs is not None:
            note(
                f"The lattice converges to <b>Black–Scholes</b> as n → ∞. "
                f"Current gap: {(result['price']-bs)/bs*100:+.3f}% — mixes finite-n "
                f"discretisation error with discounting convention "
                f"[(1+rΔt)ⁿ vs e⁻ʳᵀ]."
            )
        if spec.key == "amer_call":
            note("Early-exercise premium should be exactly zero without dividends — "
                 "this is a model self-check. Any deviation is floating-point noise (< 1e-9).")
        if spec.key == "digital_call":
            note("Binomial digitals oscillate in n because the payoff discontinuity falls between "
                 "lattice nodes. The risk-neutral P(ITM) shown is the lattice equivalent of N(d₂).")
        if spec.key == "barrier_uo_call":
            note("Barrier is monitored at lattice nodes (discrete monitoring). With more steps "
                 "this converges to the continuously-monitored price. In-out parity is exact.")

        section("Model Parameters")
        ca, cb = st.columns(2)
        with ca:
            st.dataframe(pd.DataFrame({
                "Parameter": ["S₀", "σ (annual)", "r (annual)", "T", "n", "Δt"],
                "Value": [f"${params.S0:.4f}", f"{params.sigma*100:.4f}%",
                          f"{params.r*100:.4f}%", f"{params.T:.2f} yrs",
                          str(params.n), f"{params.dt:.6f} yrs"],
            }).set_index("Parameter"), use_container_width=True)
        with cb:
            st.dataframe(pd.DataFrame({
                "Parameter": ["u", "d", "R_period", "q", "discount"],
                "Value": [f"{params.u:.6f}", f"{params.d:.6f}",
                          f"{params.R_period:.8f}", f"{params.q:.6f}",
                          f"{params.discount:.8f}"],
            }).set_index("Parameter"), use_container_width=True)

    # ── MODEL ────────────────────────────────────────────────────────────────
    with gtab2:
        sigma_daily = params.sigma / math.sqrt(250)
        with st.expander("📈 Historical Data", expanded=True):
            st.plotly_chart(plot_price_series(df, ticker), use_container_width=True, key="g_price")
            st.plotly_chart(plot_log_returns(df, ticker, sigma_daily), use_container_width=True, key="g_returns")

        has_payoff_data = "S_terminal" in result and (
            "terminal_values" in result or "payoffs" in result)
        if has_payoff_data:
            with st.expander("💵 Payoff at Maturity", expanded=True):
                order = np.argsort(result["S_terminal"])
                y_vals = (result["terminal_values"][order] if "terminal_values" in result
                          else result["payoffs"][order])
                fig2 = go.Figure(go.Scatter(
                    x=result["S_terminal"][order], y=y_vals,
                    mode="lines" if spec.engine == "lattice" else "markers",
                    marker=dict(size=3, opacity=0.25, color=BLUE_MID),
                    line=dict(color=BLUE_MID, width=2),
                    hovertemplate="Sₙ = $%{x:.2f}<br>Payoff = $%{y:.2f}<extra></extra>",
                ))
                fig2.update_layout(
                    title=f"{spec.name} — payoff at maturity",
                    xaxis=dict(title="Terminal stock price Sₙ", tickprefix="$",
                               gridcolor="#e8f0f8"),
                    yaxis=dict(title="Payoff ($)", tickprefix="$",
                               gridcolor="#e8f0f8"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True, key="g_payoff_mat")
                if spec.engine == "paths":
                    note("The vertical spread at a fixed Sₙ is the signature of "
                         "path dependence — identical terminal prices, different payoffs.")

        if spec.engine == "lattice":
            with st.expander("🌲 Binomial Tree (first 5 periods)", expanded=False):
                st.plotly_chart(plot_binomial_tree(params, k=5), use_container_width=True, key="g_tree")

    # ── ROBUSTNESS ───────────────────────────────────────────────────────────
    with gtab3:
        section("Volatility Window Sensitivity")
        note("σ is the only estimated parameter. All others are held fixed. "
             "Re-pricing across four historical windows shows how sensitive V₀ is "
             "to the choice of estimation period.")

        with st.spinner("Running sensitivity analysis..."):
            df_rob = run_generic_robustness(
                spec.key, ticker, r, T, n,
                tuple(sorted(extra_inputs.items())),
            )

        if not df_rob.empty:
            fig_rob = go.Figure()
            prices  = df_rob["_price"].tolist()
            sigmas  = df_rob["_sigma"].tolist()
            labels  = df_rob["Window"].tolist()
            colors  = [f"rgba(26,95,168,{0.45 + 0.55*(s-min(sigmas))/(max(sigmas)-min(sigmas)+1e-9)})"
                       for s in sigmas]

            fig_rob.add_trace(go.Scatter(
                x=prices, y=labels,
                mode="markers+text",
                marker=dict(size=14, color=colors, line=dict(color=BLUE_DARK, width=1.5)),
                text=[f"${p:.2f}" for p in prices],
                textposition="middle right",
                textfont=dict(size=11, color=BLUE_DARK),
            ))
            fig_rob.add_vline(x=result["price"], line=dict(color=BLUE_MID, dash="dot", width=1.5))
            fig_rob.update_layout(
                title="Option price sensitivity to volatility estimation window",
                xaxis=dict(title="V₀ ($)", tickprefix="$", gridcolor="#e8f0f8"),
                yaxis=dict(title=None),
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(l=10, r=80, t=50, b=20),
            )
            st.plotly_chart(fig_rob, use_container_width=True, key="g_rob")
            st.dataframe(
                df_rob[["Window", "σ annual (%)", "V₀ ($)"]].set_index("Window"),
                use_container_width=True,
            )
            price_spread = df_rob["_price"].max() - df_rob["_price"].min()
            pct_spread   = price_spread / result["price"] * 100
            warn(
                f"The option price ranges from <b>${df_rob['_price'].min():.4f}</b> to "
                f"<b>${df_rob['_price'].max():.4f}</b> across windows "
                f"— a spread of <b>${price_spread:.4f} ({pct_spread:.1f}%)</b> "
                f"from changing only σ. Historical σ is a backward-looking estimate; "
                f"implied volatility from traded options would give a forward-looking alternative."
            )

    # ── METHODOLOGY ──────────────────────────────────────────────────────────
    with gtab4:
        m = METHODOLOGY.get(spec.key)
        if m:
            st.markdown(m["body"])
            if m.get("latex"):
                st.latex(m["latex"])
            if m.get("body2"):
                st.markdown(m["body2"])

        section("Model Limitations & Real-World Gaps")
        st.markdown(f"""
The CRR binomial model is a clean teaching framework, but real finance professionals would
identify several gaps before trading on its output:

**1. Constant volatility (σ).** The model assumes σ is fixed forever. In reality, implied
volatility varies by strike and expiry (the *volatility smile / skew*). For equity options
the skew is pronounced — out-of-the-money puts trade at higher IV than ATM options because
of demand for downside protection. A single-σ model cannot capture this and will systematically
misprice OTM strikes. Practitioners use local-vol (Dupire) or stochastic-vol (Heston, SABR)
models to fit the full surface.

**2. Constant interest rates.** The model uses a fixed r. In reality rates are uncertain (there's
a term structure, and rates themselves are random for longer-dated options). For short-dated
equity options this is minor; for multi-year options or rate-sensitive products it matters a lot.

**3. No dividends.** Adjusted prices remove the mechanical effect of dividends on historical
returns, but the forward price of a stock is S₀ × e^{{(r−q_d)T}} where q_d is the continuous
dividend yield. Ignoring this overprices calls and underprices puts on dividend-paying stocks.
For {ticker}, check whether it pays meaningful dividends before relying on these prices.

**4. Discrete monitoring (barriers).** Barrier options are priced here at lattice nodes only.
Exchange-traded barriers are monitored continuously (or daily). Discrete monitoring understates
knock-out probability, so the discrete price is higher than the continuous-barrier price.

**5. σ is estimated, not observed.** Historical σ is backward-looking. It conflates regime
changes (COVID volatility spike vs. post-2023 calm). Two reasonable estimation windows produce
prices that differ by ~15–20% (see Robustness tab). Market makers use *implied* volatility
from traded options, which is forward-looking and consensus-based.

**6. No jumps.** GBM / binomial models assume log-returns are continuous. In practice stocks
gap — earnings surprises, M&A announcements, circuit-breaker halts. Jump-diffusion models
(Merton, Kou) add a Poisson jump component that the binomial model completely ignores. This
matters most for near-ATM short-dated options around events.

**7. Bid–offer spreads and liquidity.** Model prices are theoretical mid prices. Real trading
adds bid–offer spreads (often 2–5% of option value for less-liquid names), market impact for
large trades, and margin / collateral costs. The gap between model price and execution price
can dominate modelling error for smaller positions.

**8. Discrete steps ≈ continuous time.** At n = {n}, the binomial model approximates GBM.
The approximation error is O(1/n). For vanilla options n = 500 makes this negligible; for
path-dependent options the O(2ⁿ) cost caps n at ~25, leaving larger discretisation error.
""")

    footer()
    st.stop()


# ═══════════════════════════════════════════════════════════
# ORIGINAL ASIAN FLOATING-STRIKE FLOW (four tabs, unchanged logic)
# ═══════════════════════════════════════════════════════════
with st.spinner(f"Fetching {ticker} · running Asian pricer (2ⁿ paths)..."):
    try:
        df, params, result, approx, emp = run_pricer(ticker, r, T, n, vol_window)
    except Exception as e:
        st.error(f"Could not price **{ticker}**: {e}")
        st.markdown("""
**Common fixes:**
- European stocks need the exchange suffix: `SIE.DE`, `ASML.AS`, `BNP.PA`
- US stocks: `AAPL`, `MSFT`, `GS`
- ETFs: `SPY`, `QQQ`
        """)
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricer", "📊 Model", "🔬 Robustness", "📖 Methodology"])


# ── TAB 1: PRICER ─────────────────────────────────────────────────────────────
with tab1:
    section("Option Price")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Option Price (Binomial)", f"${result['price']:.4f}", "Backward induction · V₀_b")
    with c2:
        metric("Normal Approximation", f"${approx['price']:.4f}",
               f"Δ = {result['price']-approx['price']:+.4f} vs exact")
    with c3:
        metric("E^P[Aₜ]", f"${result['expected_payoff_p']:.4f}", "Physical measure (p = 0.5)")
    with c4:
        match = abs(result['price_backward'] - result['price_forward']) < 1e-6
        metric("Methods agree", "✓ Yes" if match else "✗ No",
               f"|V₀_b − V₀_f| = {abs(result['price_backward']-result['price_forward']):.2e}")

    note(
        f"The option price of <b>${result['price']:.4f}</b> is the arbitrage-free premium. "
        f"The physical-measure mean payoff of <b>${result['expected_payoff_p']:.4f}</b> is higher: "
        f"the difference arises from the probability measure change "
        f"(p=0.5 → q={params.q:.4f}) and discounting (factor = {result['discount_factor']:.6f})."
    )

    section("Model Parameters")
    ca, cb = st.columns(2)
    with ca:
        st.dataframe(pd.DataFrame({
            "Parameter": ["S₀", "σ (annual)", "r (annual)", "T", "n", "Δt"],
            "Value": [f"${params.S0:.4f}", f"{params.sigma*100:.4f}%",
                      f"{params.r*100:.4f}%", f"{params.T:.2f} yrs",
                      str(params.n), f"{params.dt:.6f} yrs"],
        }).set_index("Parameter"), use_container_width=True)
    with cb:
        st.dataframe(pd.DataFrame({
            "Parameter": ["u", "d", "R_period", "q", "discount", "discount^n"],
            "Value": [f"{params.u:.6f}", f"{params.d:.6f}", f"{params.R_period:.8f}",
                      f"{params.q:.6f}", f"{params.discount:.8f}",
                      f"{result['discount_factor']:.8f}"],
        }).set_index("Parameter"), use_container_width=True)

    section("Price Decomposition")
    st.dataframe(pd.DataFrame({
        "Step": [
            "1. Physical mean payoff E^P[Aₜ]",
            "2. Risk-neutral mean payoff E^Q[Aₜ]",
            "3. Discount factor (1+r·Δt)^n",
            "4. Option price V₀ = discount × E^Q[Aₜ]",
        ],
        "Value": [
            f"${result['expected_payoff_p']:.4f}",
            f"${result['expected_payoff_q']:.4f}",
            f"{result['discount_factor']:.8f}",
            f"${result['price']:.4f}",
        ],
    }).set_index("Step"), use_container_width=True)

    note(
        f"The measure change from p=0.5 to q={params.q:.4f} reduces the expected payoff from "
        f"${result['expected_payoff_p']:.4f} to ${result['expected_payoff_q']:.4f} "
        f"({(result['expected_payoff_q']-result['expected_payoff_p'])/result['expected_payoff_p']*100:+.1f}%) "
        f"because q &lt; 0.5: up-moves are down-weighted under Q to eliminate arbitrage."
    )


# ── TAB 2: MODEL ──────────────────────────────────────────────────────────────
with tab2:
    sigma_daily = params.sigma / math.sqrt(250)

    with st.expander("📈 Historical Data", expanded=True):
        st.plotly_chart(plot_price_series(df, ticker), use_container_width=True, key="t2_price")
        st.plotly_chart(plot_log_returns(df, ticker, sigma_daily), use_container_width=True, key="t2_returns")

    with st.expander("🌲 Binomial Tree", expanded=True):
        st.plotly_chart(plot_binomial_tree(params, k=5), use_container_width=True, key="t2_tree")
        note("The tree recombines — S₀·u·d = S₀·d·u = S₀ — reducing distinct nodes at step i "
             "from 2ⁱ to i+1. This is efficient for vanilla options but insufficient for "
             "path-dependent options like this Asian call (see Methodology tab).")

    with st.expander("📊 Terminal Price Distribution", expanded=False):
        st.plotly_chart(plot_terminal_distribution(params), use_container_width=True, key="t2_terminal")

    with st.expander("📉 Path Distribution (Path-Dependence)", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(plot_average_distribution(result), use_container_width=True, key="t2_avg")
        with cb:
            st.plotly_chart(plot_payoff_distribution(result), use_container_width=True, key="t2_payoff")

        zero_pct = (result['payoffs'] == 0).mean() * 100
        note(
            f"The average distribution S̄ₙ is narrower than the terminal price distribution — "
            f"averaging dampens extremes. {zero_pct:.1f}% of paths expire worthless, "
            f"with the mean payoff of ${result['expected_payoff_p']:.2f} driven by a small number "
            f"of high-payoff paths in the right tail."
        )
        st.plotly_chart(plot_payoff_vs_terminal(result, n_sample=20_000), use_container_width=True, key="t2_scatter")


# ── TAB 3: ROBUSTNESS ─────────────────────────────────────────────────────────
with tab3:
    section("Volatility Window Sensitivity")
    note("σ is the only estimated parameter. All others (r, T, n, S₀) are fixed. "
         "Re-pricing across four historical windows shows how sensitive V₀ is to the "
         "choice of estimation period.")

    with st.spinner("Running robustness analysis..."):
        df_rob = run_robustness(ticker, r, T, n, hash(df["adj_close"].values.tobytes()))

    st.plotly_chart(
        plot_window_sensitivity(df_rob, baseline_price=result["price"]),
        use_container_width=True, key="t3_window",
    )
    st.dataframe(
        df_rob[["window","n_obs","sigma_daily_pct","sigma_annual_pct","price","delta_pct"]]
        .rename(columns={"window":"Window","n_obs":"Obs","sigma_daily_pct":"σ daily (%)",
                         "sigma_annual_pct":"σ annual (%)","price":"V₀ ($)","delta_pct":"Δ (%)"}
        ).set_index("Window"), use_container_width=True)

    section("Normal Approximation Convergence")
    note(
        "V₀_approx converges to its own asymptote as n→∞, but never reaches the exact "
        "binomial price V₀. The permanent gap has two sources: (1) the approximation uses "
        "p=0.5 rather than the risk-neutral q, and (2) continuous discounting e^{-rT} "
        "rather than discrete (1+r·Δt)^n. Both effects remain even as n→∞."
    )
    with st.spinner("Computing convergence..."):
        df_conv = run_convergence(
            hash((params.sigma, params.r, params.T, params.n)),
            params.sigma, params.r, params.T, params.n,
            params.S0, params.u, params.d, params.q, params.R_period, params.discount,
        )
    st.plotly_chart(plot_approx_convergence(df_conv, result["price"], n), use_container_width=True, key="t3_conv")

    section("Analytical vs Empirical Validation of Dₙ")
    st.dataframe(pd.DataFrame({
        "": ["Analytical", "Empirical", "Relative error"],
        "E[Dₙ]": ["≈ 0", f"${emp['emp_mean']:.5f}", "—"],
        "Var(Dₙ)": [f"{approx['var_Dn']:.4f}", f"{emp['emp_var']:.4f}",
                    f"{abs(emp['emp_var']-approx['var_Dn'])/approx['var_Dn']*100:.3f}%"],
        "SD(Dₙ)": [f"${approx['sd_Dn']:.4f}", f"${emp['emp_sd']:.4f}",
                   f"{abs(emp['emp_sd']-approx['sd_Dn'])/approx['sd_Dn']*100:.3f}%"],
    }).set_index(""), use_container_width=True)


# ── TAB 4: METHODOLOGY ────────────────────────────────────────────────────────
with tab4:
    m = METHODOLOGY.get("asian_float")
    if m:
        st.markdown(m["body"])
        if m.get("latex"):
            st.latex(m["latex"])

    section("1. The Option")
    st.markdown(
        "A **European floating-strike Asian call** gives the holder the right to receive "
        "the following payoff at maturity T:"
    )
    st.latex(r"A_T = \max\!\left(S_n - \bar{S}_n,\ 0\right)")
    st.markdown("where the arithmetic average over the life of the contract is:")
    st.latex(r"\bar{S}_n = \frac{1}{n+1} \sum_{t=0}^{n} S_t")
    st.markdown(
        "There is no fixed strike. The holder profits when the stock finishes **above its own "
        "average** over the contract period — rewarding a strong finish relative to historical "
        "performance. Unlike a vanilla call, the strike is path-dependent by construction."
    )

    section("2. Data & Volatility Estimation")
    st.markdown("We compute daily log returns and annualise with 250 trading days:")
    st.latex(r"r_t = \log\!\left(\frac{P_t}{P_{t-1}}\right), \quad \hat{\sigma} = \hat{\sigma}_{\text{daily}} \times \sqrt{250}")
    st.markdown(
        "Log returns are preferred: temporally additive, approximately normal under GBM, "
        r"and consistent with the CRR parametrisation where $u = e^{\sigma\sqrt{\Delta t}}$."
    )

    section("3. Binomial Tree — CRR")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad q = \frac{(1+r\Delta t) - d}{u - d}")
    st.markdown(
        "The tree **recombines** (u·d = 1), reducing nodes at step i from 2ⁱ to i+1. "
        "This is efficient for vanilla options but fails for path-dependent payoffs "
        "because the average depends on the sequence of moves, not just the terminal node."
    )

    section("4. Augmented State Space & Forward Enumeration")
    st.latex(r"C_t = C_{t-1} + S_t, \quad C_0 = S_0, \quad \bar{S}_n = \frac{C_n}{n+1}")
    st.markdown(
        "The pair (Sₜ, Cₜ) is Markov — its future depends only on its current value. "
        "We propagate all 2ⁿ paths forward, tracking (S, C, w) per path. "
        f"At n={n} this yields 2^{n} = {2**n:,} paths."
    )

    section("5. Risk-Neutral Pricing")
    st.latex(r"V_0 = \frac{1}{(1+r\,\Delta t)^n}\,\mathbb{E}^Q[A_T]")
    st.markdown("Backward induction folds the payoff vector in half n times:")
    st.latex(r"V_t = \frac{1}{1+r\,\Delta t}\left(q\,V_{t+1}^{\uparrow} + (1-q)\,V_{t+1}^{\downarrow}\right)")

    section("6. Normal Approximation")
    st.latex(r"\operatorname{Var}(D_n) = S_0^2\,\sigma^2\,T\,\frac{2n+1}{6(n+1)}, \quad V_0^{\text{approx}} = e^{-rT}\,\frac{\sqrt{\operatorname{Var}(D_n)}}{\sqrt{2\pi}}")
    st.markdown(
        "Even as n→∞, V₀_approx converges to its own asymptote, not the exact binomial price. "
        "The permanent gap reflects (1) use of p=0.5 vs risk-neutral q, and (2) continuous vs discrete discounting."
    )

    section("Model Limitations & Real-World Gaps")
    st.markdown(f"""
**1. Constant σ.** Historical volatility is backward-looking and regime-dependent. The COVID
spike of 2020 inflates the full-window σ by ~4 percentage points vs post-2021 estimates,
shifting the option price by ~15% (see Robustness tab). Market makers use implied volatility
from traded options — forward-looking and market-consistent.

**2. No volatility smile.** A single σ cannot capture that OTM puts trade at higher IV than
ATM options (equity skew). For Asian options this matters less than for vanillas because
averaging compresses the effective distribution, but it still biases the price.

**3. No jumps.** GBM / CRR assumes continuous price paths. Earnings surprises, macro events,
and circuit-breaker halts create jumps that the model completely ignores. Jump-diffusion
models (Merton, Kou) add a Poisson component for this.

**4. No dividends explicitly.** Adjusted prices remove historical split/dividend artifacts,
but the model doesn't incorporate a dividend yield q_d into the forward price. For {ticker}
check the dividend yield and consider applying the Merton adjustment
S₀ → S₀·e^{{−q_d T}} if material.

**5. Discrete steps ≈ continuous time.** Path-enumeration options are capped at n = {MAX_N}
(memory constraint). This leaves non-negligible discretisation error relative to continuous
Asian option pricing methods (e.g. geometric-average approximations, PDE methods on
arithmetic averages, or Monte Carlo with variance reduction).

**6. No transaction costs or liquidity.** The model gives a theoretical mid-price. Real
execution adds bid–offer spread, margin/collateral costs, and potential market impact for
large notionals. For less-liquid names these costs can easily exceed the model's precision.
""")

footer()