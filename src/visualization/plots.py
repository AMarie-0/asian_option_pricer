"""
plots.py — all Plotly visualisations for the Streamlit app.
Each function returns a go.Figure ready for st.plotly_chart().

Key design decision: the average/payoff distribution charts pre-compute
histogram bins in numpy and pass only ~60 counts to Plotly — NOT the raw
33M path values — to stay well under Streamlit's 200MB message limit.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_BLUE  = "#1a3a5c"
MID_BLUE   = "#2e6da4"
LIGHT_BLUE = "#6baed6"
LIGHT_GREY = "#d4d4d4"
RED        = "#c0392b"
BG         = "rgba(0,0,0,0)"

def _base(title="", height=None, **kw) -> dict:
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#222"),
        margin=dict(l=50, r=30, t=50, b=50),
        title=title,
    )
    if height:
        d["height"] = height
    d.update(kw)
    return d


# ── 1. Historical price series ────────────────────────────────────────────────

def plot_price_series(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["adj_close"],
        mode="lines", line=dict(color=DARK_BLUE, width=1.5), name=ticker,
    ))
    fig.update_layout(**_base(f"{ticker} — Adjusted Closing Price"),
                      xaxis_title="Date", yaxis_title="Price (USD)",
                      showlegend=False)
    fig.update_yaxes(tickprefix="$")
    return fig


# ── 2. Daily log returns ──────────────────────────────────────────────────────

def plot_log_returns(df: pd.DataFrame, ticker: str, sigma_daily: float) -> go.Figure:
    prices  = df["adj_close"].dropna().values
    returns = np.diff(np.log(prices))
    dates   = df["date"].iloc[1:].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=returns, mode="lines",
        line=dict(color=MID_BLUE, width=0.8), name="Log return",
    ))
    for sign, pos in [(1, "top right"), (-1, "bottom right")]:
        label = f"+1σ = {sigma_daily*100:.2f}%" if sign > 0 else f"−1σ = {sigma_daily*100:.2f}%"
        fig.add_hline(y=sign * sigma_daily,
                      line=dict(color=RED, dash="dash", width=1),
                      annotation_text=label, annotation_position=pos)
    fig.update_layout(**_base(f"{ticker} — Daily Log Returns"),
                      xaxis_title="Date", yaxis_title="Log Return",
                      yaxis_tickformat=".1%", showlegend=False)
    return fig


# ── 3. Binomial tree (first k periods) ───────────────────────────────────────

def plot_binomial_tree(p, k: int = 5) -> go.Figure:
    from src.model.binomial_tree import build_stock_matrix
    mat = build_stock_matrix(p)
    k   = min(k, p.n)

    edge_x, edge_y = [], []
    node_x, node_y, node_text, node_color = [], [], [], []

    for i in range(k + 1):
        for j in range(i + 1):
            price = mat[i, j]
            is_recomb = (i == 2 and j == 1)
            node_x.append(i); node_y.append(j)
            node_text.append(f"${price:.2f}")
            node_color.append(RED if is_recomb else MID_BLUE)
            if i < k:
                for dj in [0, 1]:
                    edge_x += [i, i + 1, None]
                    edge_y += [j, j + dj, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=LIGHT_GREY, width=1), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=36, color=node_color),
        text=node_text,
        textfont=dict(color="white", size=9),
        textposition="middle center", hoverinfo="text",
    ))
    fig.update_layout(
        **_base(f"Binomial Tree — First {k} Periods", height=350),
        xaxis=dict(title="Time step",
                   tickvals=list(range(k + 1)),
                   ticktext=[f"t={i}" for i in range(k + 1)],
                   showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        annotations=[dict(
            text="Red node = recombination: S₀·u·d = S₀·d·u = S₀",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=10, color=RED),
        )],
    )
    return fig


# ── 4. Terminal price distribution (physical vs risk-neutral) ─────────────────

def plot_terminal_distribution(p) -> go.Figure:
    from math import comb
    j_vals   = np.arange(p.n + 1)
    prices_T = p.S0 * p.u ** j_vals * p.d ** (p.n - j_vals)
    probs_p  = np.array([comb(p.n, int(j)) * 0.5 ** p.n for j in j_vals])
    probs_q  = np.array([
        comb(p.n, int(j)) * (p.q ** j) * ((1 - p.q) ** (p.n - j))
        for j in j_vals
    ])
    width = np.diff(prices_T).mean() * 0.8

    fig = go.Figure()
    fig.add_trace(go.Bar(x=prices_T, y=probs_p, width=width,
                         name="Physical (p=0.5)", marker_color=LIGHT_GREY, opacity=0.85))
    fig.add_trace(go.Bar(x=prices_T, y=probs_q, width=width * 0.5,
                         name="Risk-neutral (q)", marker_color=RED, opacity=0.7))
    fig.add_vline(x=p.S0, line=dict(color=DARK_BLUE, dash="dash"),
                  annotation_text=f"S₀ = ${p.S0:.2f}", annotation_position="top right")
    fig.update_layout(
        **_base("Terminal Price Distribution: Physical vs Risk-Neutral"),
        xaxis_title="Terminal Stock Price ($)", xaxis_tickprefix="$",
        yaxis_title="Probability", yaxis_tickformat=".1%",
        barmode="overlay", legend=dict(x=0.6, y=0.95),
    )
    return fig


# ── 5. Average distribution (pre-binned — avoids 200MB Streamlit limit) ──────

def plot_average_distribution(result: dict) -> go.Figure:
    S_bar    = result["S_bar"]
    counts, edges = np.histogram(S_bar, bins=60)
    centres  = (edges[:-1] + edges[1:]) / 2
    bin_w    = edges[1] - edges[0]
    density  = counts / (counts.sum() * bin_w)
    mean_val = float(np.mean(S_bar))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=density, width=bin_w * 0.9,
        marker_color=MID_BLUE, opacity=0.8, name="S̄ₙ",
    ))
    fig.add_vline(x=mean_val, line=dict(color=RED, dash="dash"),
                  annotation_text=f"Mean = ${mean_val:.2f}",
                  annotation_position="top right")
    fig.update_layout(
        **_base("Distribution of Arithmetic Averages at Maturity"),
        xaxis_title="Arithmetic Average Price ($)", xaxis_tickprefix="$",
        yaxis_title="Density", showlegend=False,
    )
    return fig


# ── 6. Payoff distribution (pre-binned, zero spike separated) ────────────────

def plot_payoff_distribution(result: dict) -> go.Figure:
    payoffs     = result["payoffs"]
    mean_val    = float(np.mean(payoffs))
    zero_pct    = float((payoffs == 0).mean()) * 100
    pos_payoffs = payoffs[payoffs > 0]

    counts, edges = np.histogram(pos_payoffs, bins=55)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_w   = edges[1] - edges[0]
    # density relative to ALL paths (including zeros)
    density = counts / (len(payoffs) * bin_w)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[0], y=[zero_pct / 100 / bin_w],
        width=[bin_w * 0.9],
        marker_color=DARK_BLUE, opacity=0.85,
        name=f"Zero payoff ({zero_pct:.1f}%)",
    ))
    fig.add_trace(go.Bar(
        x=centres, y=density, width=bin_w * 0.9,
        marker_color=MID_BLUE, opacity=0.75,
        name="Positive payoff",
    ))
    fig.add_vline(x=mean_val, line=dict(color=RED, dash="dash"),
                  annotation_text=f"Mean = ${mean_val:.2f}",
                  annotation_position="top right")
    fig.update_layout(
        **_base("Distribution of Floating-Strike Payoffs at Maturity"),
        xaxis_title="Payoff ($)", xaxis_tickprefix="$",
        yaxis_title="Density",
        barmode="overlay", legend=dict(x=0.55, y=0.9),
    )
    return fig


# ── 7. Payoff vs terminal price ───────────────────────────────────────────────

def plot_payoff_vs_terminal(result: dict, n_sample: int = 80_000) -> go.Figure:
    S   = result["S_terminal"]
    pay = result["payoffs"]
    idx = np.random.choice(len(S), size=min(n_sample, len(S)), replace=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S[idx], y=pay[idx], mode="markers",
        marker=dict(color=RED, size=2, opacity=0.25),
        hoverinfo="skip",
    ))
    fig.update_layout(
        **_base("Payoff vs Terminal Stock Price"),
        xaxis_title="Terminal Stock Price Sₙ ($)", xaxis_tickprefix="$",
        yaxis_title="Payoff ($)", yaxis_tickprefix="$",
        showlegend=False,
        annotations=[dict(
            text="Vertical spread = path dependence: same Sₙ, different averages → different payoffs",
            xref="paper", yref="paper", x=0.5, y=1.06,
            showarrow=False, font=dict(size=10, color="grey"),
        )],
    )
    return fig


# ── 8. Robustness — window sensitivity (clean horizontal bars) ───────────────

def plot_window_sensitivity(df_rob: pd.DataFrame, baseline_price: float) -> go.Figure:
    colors = [DARK_BLUE, MID_BLUE, LIGHT_BLUE, "#bdd7e7"]
    fig = go.Figure()

    for i, (_, row) in enumerate(df_rob.iterrows()):
        fig.add_trace(go.Bar(
            x=[row["price"]], y=[row["window"]],
            orientation="h", width=0.5,
            marker_color=colors[i % len(colors)],
            marker_line_width=0,
            showlegend=False,
            hovertemplate=(
                f"<b>{row['window']}</b><br>"
                f"V₀ = ${row['price']:.4f}<br>"
                f"σ = {row['sigma_annual_pct']:.2f}%<br>"
                f"Δ = {row['delta_pct']:+.2f}%"
                "<extra></extra>"
            ),
        ))
        fig.add_annotation(
            x=row["price"] + baseline_price * 0.004,
            y=row["window"],
            text=f"<b>${row['price']:.2f}</b>  σ={row['sigma_annual_pct']:.1f}%  ({row['delta_pct']:+.1f}%)",
            showarrow=False, xanchor="left",
            font=dict(size=11, color="#222"),
        )

    fig.add_vline(x=baseline_price,
                  line=dict(color=RED, dash="dash", width=1.5),
                  annotation_text="Baseline",
                  annotation_position="top left",
                  annotation_font_color=RED)

    x_min = df_rob["price"].min() * 0.96
    x_max = df_rob["price"].max() * 1.10

    fig.update_layout(
        **_base("Option Price Sensitivity to Volatility Estimation Window", height=280),
        xaxis=dict(title="Option Price V₀ ($)", tickprefix="$",
                   range=[x_min, x_max], showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        barmode="overlay",
    )
    return fig


# ── 9. Normal approximation convergence ──────────────────────────────────────

def plot_approx_convergence(df_conv: pd.DataFrame, exact_price: float, n_baseline: int = 25) -> go.Figure:
    asymptote = float(df_conv["price_limit"].iloc[-1])
    gap       = exact_price - asymptote

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_conv["n"], y=df_conv["price_approx"],
        mode="lines+markers",
        line=dict(color=DARK_BLUE, width=2), marker=dict(size=5),
        name="V₀ approx(n)",
    ))
    fig.add_trace(go.Scatter(
        x=df_conv["n"], y=df_conv["price_limit"],
        mode="lines", line=dict(color=LIGHT_GREY, dash="dash", width=1.5),
        name=f"Asymptote n→∞  (${asymptote:.2f})",
    ))
    fig.add_hline(
        y=exact_price,
        line=dict(color=RED, dash="dot", width=1.5),
        annotation_text=f"Exact V₀ = ${exact_price:.2f}  (binomial, risk-neutral q)",
        annotation_position="top right",
        annotation_font_color=RED,
    )

    # Highlight n=baseline
    row_n = df_conv[df_conv["n"] == n_baseline]
    if not row_n.empty:
        approx_val = float(row_n["price_approx"].iloc[0])
        fig.add_trace(go.Scatter(
            x=row_n["n"], y=row_n["price_approx"],
            mode="markers", marker=dict(color=RED, size=10),
            name=f"n={n_baseline}  (${approx_val:.2f})",
        ))

    # Annotate the permanent gap
    mid_y   = (exact_price + asymptote) / 2
    max_n   = int(df_conv["n"].max())
    fig.add_annotation(
        x=max_n * 0.75, y=mid_y,
        text=(f"Permanent gap = ${gap:.2f}<br>"
              f"Source: p=0.5 used instead of q={df_conv.get('q', [None])[0] or '≠0.5'}<br>"
              f"and e⁻ʳᵀ vs (1+r·Δt)ⁿ discounting"),
        showarrow=True, arrowhead=2, ax=50, ay=0,
        font=dict(size=10, color="#555"),
        bgcolor="white", bordercolor="#ccc", borderwidth=1,
        xanchor="left",
    )

    fig.update_layout(
        **_base("Normal Approximation — Convergence as n → ∞"),
        xaxis_title="Number of periods n",
        yaxis_title="Option price ($)", yaxis_tickprefix="$",
        legend=dict(x=0.02, y=0.25),
    )
    return fig
