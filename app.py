"""
app.py — Asian Option Pricer
Streamlit app: floating-strike Asian call option on any equity ticker.
"""
import math
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Asian Option Pricer",
    page_icon="📈",
    layout="wide",
)

from data.fetch import fetch_prices, fetch_risk_free_rate, SUPPORTED_TICKERS
from data.database import save_prices, load_prices
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

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box { background:#f7f9fc; border-radius:8px; padding:16px 20px; margin-bottom:8px; }
  .metric-label { font-size:12px; color:#666; text-transform:uppercase; letter-spacing:.05em; }
  .metric-value { font-size:28px; font-weight:700; color:#1a3a5c; margin-top:4px; }
  .metric-sub   { font-size:11px; color:#999; margin-top:2px; }
  .section-header { font-size:13px; font-weight:600; color:#2e6da4;
                    text-transform:uppercase; letter-spacing:.08em;
                    border-bottom:2px solid #2e6da4; padding-bottom:4px;
                    margin:24px 0 12px; }
  .note-box { background:#eef4fb; border-left:3px solid #2e6da4;
              padding:10px 14px; border-radius:4px;
              font-size:13px; color:#333; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

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


# ── cached data fetching ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(ticker: str) -> pd.DataFrame:
    """Fetch and cache prices for 1 hour."""
    try:
        return load_prices(ticker, db_path="data/prices.duckdb")
    except Exception:
        df = fetch_prices(ticker)
        try:
            save_prices(df, ticker, db_path="data/prices.duckdb")
        except Exception:
            pass
        return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_risk_free_rate() -> float:
    return fetch_risk_free_rate()


# ── cached computation ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pricer(ticker, r, T, n, vol_window):
    """Cache the full pricing computation — 33M paths only run once per param set."""
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
def run_convergence(_params_hash, sigma, r, T, n, S0, u, d, q, R_period, discount):
    """Cache convergence — reconstruct params from primitives (hashable)."""
    from src.model.calibration import ModelParams
    import dataclasses
    p = ModelParams("", S0, sigma, u, d, q, r, n, T/n, T, R_period, discount)
    return normal_approx_convergence(p)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Asian Option Pricer")
    st.caption("European floating-strike Asian call via multi-period binomial tree")
    st.divider()

    ticker_mode = st.radio("Ticker input", ["Curated list", "Any ticker"], horizontal=True)
    if ticker_mode == "Curated list":
        ticker = st.selectbox("Ticker", list(SUPPORTED_TICKERS.keys()), index=0)
        st.caption(SUPPORTED_TICKERS[ticker])
    else:
        ticker = st.text_input("Enter ticker symbol", value="AAPL").upper().strip()
        if ticker:
            st.caption("US: `AAPL` `MSFT` `GS` — Europe: `SIE.DE` `ASML.AS` `BNP.PA` — use Yahoo Finance suffix")

    st.subheader("Model Parameters")
    T = st.slider("Maturity T (years)", 0.25, 2.0, 0.5, 0.25)
    n = st.slider("Binomial steps n", 5, 25, 25, 5)

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
    st.caption("CRR binomial tree · 2ⁿ path enumeration · backward induction · "
               "direct forward calculation cross-check · normal approximation")


# ── main ──────────────────────────────────────────────────────────────────────
st.title("European Floating-Strike Asian Call Option")
st.latex(r"A_T = \max\!\left(S_n - \bar{S}_n,\ 0\right), \quad \bar{S}_n = \frac{1}{n+1}\sum_{t=0}^{n} S_t")

if not run:
    st.info("Configure parameters in the sidebar and click **Price** to run.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()

with st.spinner(f"Fetching {ticker} and running pricer..."):
    try:
        df, params, result, approx, emp = run_pricer(ticker, r, T, n, vol_window)
    except Exception as e:
        st.error(f"Could not price **{ticker}**: {e}")
        st.markdown("""
**Common fixes:**
- European stocks need the exchange suffix: `SIE.DE` (Siemens), `ASML.AS` (ASML), `BNP.PA` (BNP Paribas)
- Check the exact symbol on [Yahoo Finance](https://finance.yahoo.com/lookup)
- US stocks use plain tickers: `AAPL`, `MSFT`, `GS`
- ETFs: `SPY`, `QQQ`
        """)
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricer", "📊 Model", "🔬 Robustness", "📖 Methodology"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — PRICER
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# TAB 2 — MODEL
# ═══════════════════════════════════════════════════════════
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
            f"averaging dampens extremes. The payoff distribution has "
            f"{zero_pct:.1f}% of paths expiring worthless, "
            f"with the mean payoff of ${result['expected_payoff_p']:.2f} driven by a small number "
            f"of high-payoff paths in the right tail."
        )
        st.plotly_chart(plot_payoff_vs_terminal(result, n_sample=20_000), use_container_width=True, key="t2_scatter")


# ═══════════════════════════════════════════════════════════
# TAB 3 — ROBUSTNESS
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ═══════════════════════════════════════════════════════════
with tab4:
    st.caption("All charts referenced below are available in the Model and Robustness tabs.")

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
    st.markdown(
        "The model is calibrated on historical daily adjusted closing prices of the underlying "
        "equity (see **Model → Historical Data**). We compute daily log returns:"
    )
    st.latex(r"r_t = \log\!\left(\frac{P_t}{P_{t-1}}\right)")
    st.markdown("Daily volatility is estimated as the sample standard deviation:")
    st.latex(r"\hat{\sigma}_{\text{daily}} = \sqrt{\frac{1}{N-1}\sum_{t=1}^{N}(r_t - \bar{r})^2}")
    st.markdown("Annualised with 250 trading days:")
    st.latex(r"\hat{\sigma} = \hat{\sigma}_{\text{daily}} \times \sqrt{250}")
    st.markdown(
        "Log returns are preferred over simple returns for three reasons: temporal additivity "
        "(they sum across periods), approximate normality under GBM, and consistency with the "
        "CRR parametrisation where $u = e^{\sigma\sqrt{\Delta t}}$."
    )

    section("3. Binomial Tree Construction")
    st.markdown("The Cox-Ross-Rubinstein (CRR) parametrisation:")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad \Delta t = \frac{T}{n}")
    st.markdown(
        "The tree **recombines** since $u \cdot d = 1$, meaning an up-move followed by a "
        "down-move returns to the same price as the reverse sequence:"
    )
    st.latex(r"S_0 \cdot u \cdot d = S_0 \cdot d \cdot u = S_0")
    st.markdown(
        "This reduces distinct nodes at step $i$ from $2^i$ to $i+1$, making the tree "
        "tractable for standard options. See **Model → Binomial Tree** for the visualisation."
    )

    section("4. Why the Standard Tree Fails Here")
    st.markdown(
        "For a vanilla European call, the payoff $\max(S_n - K, 0)$ depends only on the "
        "terminal price $S_n$, which is fully determined by the number of up-moves $j$. "
        "All paths with the same $j$ produce the same payoff — so they can be grouped using "
        "the binomial coefficient, giving an $O(n)$ computation."
    )
    st.markdown(
        "For the floating-strike Asian call, two paths with identical $j$ — and therefore "
        "identical $S_n$ — can produce **different payoffs** if the order of moves differs, "
        "because the intermediate prices change and so does the arithmetic average. "
        "This is visible in the **Model → Payoff vs Terminal Price** scatter: at any fixed "
        "$S_n$, there is a vertical spread of payoffs from different paths. "
        "The recombining tree, which assigns a single value per node, is fundamentally insufficient."
    )

    section("5. Augmented State Space & Forward Enumeration")
    st.markdown(
        "The key insight is that we do not need to store the full path history. "
        "It suffices to track the **cumulative sum** $C_t = \sum_{s=0}^{t} S_s$ alongside "
        "the stock price, since it satisfies a simple one-step recursion:"
    )
    st.latex(r"C_t = C_{t-1} + S_t, \quad C_0 = S_0")
    st.markdown(
        "The pair $(S_t, C_t)$ is a **Markov process** — its future depends only on its "
        "current value, not the full path history. At maturity the average is recovered as:"
    )
    st.latex(r"\bar{S}_n = \frac{C_n}{n+1}")
    st.markdown(
        "We enumerate all $2^n$ paths by propagating three numpy arrays $(S, C, w)$ forward, "
        "branching at each step into $[\text{up} \mid \text{down}]$ — one entry per path. "
        "At $n=25$ this yields exactly $2^{25} = 33{,}554{,}432$ paths. "
        "See **Model → Path Distribution** for the resulting payoff and average distributions."
    )

    section("6. Risk-Neutral Pricing & Backward Induction")
    st.markdown(
        "Under the physical probability $p = 0.5$, the stock's expected return is zero "
        "while the risk-free bond returns $r > 0$ — creating an arbitrage opportunity. "
        "We price under the **risk-neutral probability** $q$ derived from the no-arbitrage condition:"
    )
    st.latex(r"R_{\text{period}} = 1 + r\,\Delta t, \qquad q = \frac{R_{\text{period}} - d}{u - d}")
    st.markdown("The arbitrage-free price is the discounted risk-neutral expectation:")
    st.latex(r"V_0 = \frac{1}{(1+r\,\Delta t)^n}\,\mathbb{E}^Q[A_T] = \frac{1}{(1+r\,\Delta t)^n} \sum_\omega q^{j(\omega)}(1-q)^{n-j(\omega)}\max\!\left(S_n(\omega)-\bar{S}_n(\omega),0\right)")
    st.markdown(
        "The $[\text{up} \mid \text{down}]$ concatenation order in the forward pass means "
        "that at every backward step, position $i$ in the first half and position $i$ in the "
        "second half are the up/down children of the same parent. Backward induction then folds "
        "the $2^n$ payoff vector in half $n$ times with pure vectorised arithmetic:"
    )
    st.latex(r"V_t = \frac{1}{1+r\,\Delta t}\left(q\,V_{t+1}^{\uparrow} + (1-q)\,V_{t+1}^{\downarrow}\right)")
    st.markdown(
        "We cross-check by recovering the up-move count algebraically from each terminal price "
        "and pricing directly — both methods must agree to floating-point precision "
        "(verified in the **Pricer → Methods Agree** indicator)."
    )
    st.latex(r"j = \operatorname{round}\!\left(\frac{\log(S/S_0) - n\log d}{\log u - \log d}\right), \qquad V_0^f = \frac{1}{(1+r\Delta t)^n}\sum_\omega q^{j(\omega)}(1-q)^{n-j(\omega)} A_T(\omega)")

    section("7. Normal Approximation")
    st.markdown(
        "As an independent cross-check we approximate $D_n = S_n - \bar{S}_n$ as normally "
        "distributed. Linearising $S_t \approx S_0(1 + \sigma\sqrt{\Delta t}\,W_t)$ and "
        "using the covariance structure of the random walk $W_t$, the analytical variance is:"
    )
    st.latex(r"\operatorname{Var}(D_n) = S_0^2\,\sigma^2\,T\,\frac{2n+1}{6(n+1)} \xrightarrow{n\to\infty} \frac{S_0^2\sigma^2 T}{3}")
    st.markdown(
        "With $E[D_n] \approx 0$ under $p=0.5$, the expected positive part of a $N(0,v^2)$ "
        "variable is $v/\sqrt{2\pi}$, giving the closed-form price:"
    )
    st.latex(r"V_0^{\text{approx}} = e^{-rT}\,\frac{\sqrt{\operatorname{Var}(D_n)}}{\sqrt{2\pi}}")
    st.markdown(
        "Even as $n \to \infty$, $V_0^{\text{approx}}$ converges to its own asymptote "
        "rather than to the exact binomial price. The **permanent gap** has two sources:"
    )
    st.markdown("""
- **Probability measure**: the approximation uses $p=0.5$ rather than the risk-neutral $q$
- **Discounting**: continuous $e^{-rT}$ rather than discrete $(1+r\Delta t)^n$

Both effects persist regardless of $n$. See **Robustness → Normal Approximation Convergence** for the chart.
""")

    section("Model Limitations")
    st.markdown("""
- **σ is estimated, not observed.** The price ranges from ~15% depending on the estimation window
  (see **Robustness → Volatility Window Sensitivity**). Implied volatility from traded options
  would give a forward-looking, market-consistent alternative.
- **Discrete approximation.** At $n=25$ the binomial model approximates continuous GBM.
  Convergence of the normal approximation is confirmed in the Robustness tab.
- **No dividends.** Adjusted prices correct for splits but the model does not explicitly
  model the continuous dividend yield $q_d$.
- **Simple discrete discounting.** We use $R = 1 + r\Delta t$ matching the standard
  discrete binomial convention. The normal approximation uses $e^{-rT}$ (continuous limit),
  which explains part of the gap between the two methods.
""")