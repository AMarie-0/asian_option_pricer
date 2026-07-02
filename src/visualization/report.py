# src/visualization/report.py
"""
One-page overnight pricing report — Deutsche Bank / Goldman Sachs style.
A4 portrait, print-ready (150 dpi).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from src.model.calibration import ModelParams

# ── Colour palette ────────────────────────────────────────────────
_BLUE  = "#003399"   # Deutsche Bank blue
_GOLD  = "#C8960C"   # accent / price marker
_DARK  = "#1A1A2A"
_GREY  = "#F2F4F8"
_MID   = "#C0C4CC"
_RED   = "#CC2200"
_WHITE = "#FFFFFF"


# ── Private helpers ───────────────────────────────────────────────

def _style_chart(ax: plt.Axes) -> None:
    ax.set_facecolor(_WHITE)
    for sp in ax.spines.values():
        sp.set_color(_MID)
        sp.set_linewidth(0.5)
    ax.grid(True, color=_GREY, lw=0.5, axis="y", zorder=0)
    ax.tick_params(labelsize=5.5, colors=_DARK, length=2)


def _draw_header(ax: plt.Axes, ticker: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0, 0.95, "EQUITY DERIVATIVES  ·  RESEARCH",
            fontsize=6.5, color=_BLUE, fontweight="bold", va="top")
    ax.text(0, 0.60,
            f"{ticker}  —  Asian Call Option  ·  Overnight Pricing Report",
            fontsize=13, color=_DARK, fontweight="bold", va="top")
    ax.text(0, 0.18,
            "Floating-Strike  ·  European-Style  ·  Binomial Tree (Cox-Ross-Rubinstein, CRR)",
            fontsize=7, color="#666666", va="top")
    ax.text(1, 0.95, date.today().strftime("%d %B %Y"),
            fontsize=7, color="#666666", va="top", ha="right")
    # thick blue rule at bottom of header
    ax.axhline(0, color=_BLUE, lw=2.5)


def _draw_price_box(ax: plt.Axes, price: float, approx: float, n_states: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.04, 0.06), 0.92, 0.88,
        boxstyle="round,pad=0.02", lw=1.8,
        edgecolor=_BLUE, facecolor=_GREY,
    )
    ax.add_patch(box)

    ax.text(0.5, 0.94, "OPTION PRICE", fontsize=7.5, fontweight="bold",
            color=_BLUE, ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.75, f"${price:.4f}", fontsize=28, fontweight="bold",
            color=_DARK, ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.50, "Binomial Tree", fontsize=7.5, color="#555555",
            ha="center", va="top", transform=ax.transAxes)

    ax.axhline(0.43, xmin=0.12, xmax=0.88, color=_MID, lw=0.9)

    ax.text(0.5, 0.38, f"${approx:.4f}  ·  Normal Approximation",
            fontsize=7.5, color="#444444", ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.19, f"{n_states:,} terminal states",
            fontsize=6.5, color="#999999", ha="center", va="top", transform=ax.transAxes)


def _draw_params_table(ax: plt.Axes, params: ModelParams, window: str) -> None:
    ax.axis("off")
    r_daily = params.r / 250
    sig_d   = params.sigma / math.sqrt(250)

    rows = [
        ("Initial price",      f"${params.S0:,.2f}",                       "Dataset"),
        ("Maturity",           f"{params.T:.1f} years",                    "Input"),
        ("Periods",            f"{params.n}",                              "Input"),
        ("Risk-free (annual)", f"{params.r * 100:.2f}% p.a.",             "Input"),
        ("Risk-free (daily)",  f"{r_daily * 100:.4f}% / day",             "Derived"),
        ("Period length Δt",   f"{params.dt:.4f} yrs",                    "Derived"),
        ("Daily volatility",   f"{sig_d * 100:.3f}%",                     f"Est. [{window}]"),
        ("Annual volatility",  f"{params.sigma * 100:.2f}%",              f"Est. [{window}]"),
        ("Up factor u",        f"{params.u:.4f}",                         "Derived"),
        ("Down factor d",      f"{params.d:.4f}",                         "Derived"),
        ("RN probability q",   f"{params.q:.4f}",                         "Derived"),
        ("Physical prob. p",   "0.5000",                                   "Assigned"),
    ]

    ax.text(0, 1.03, "MODEL PARAMETERS", fontsize=7, fontweight="bold",
            color=_BLUE, transform=ax.transAxes, va="bottom")

    col_x = [0.0, 0.50, 0.82]
    for hdr, x in zip(["Parameter", "Value", "Source"], col_x):
        ax.text(x, 0.98, hdr, fontsize=6.5, fontweight="bold",
                color=_DARK, transform=ax.transAxes, va="top")
    ax.plot([0, 1], [0.964, 0.964], color=_BLUE, lw=0.7, transform=ax.transAxes)

    y, step = 0.942, 0.074
    for label, val, src in rows:
        ax.text(col_x[0], y, label, fontsize=6.2, color=_DARK,
                transform=ax.transAxes, va="top")
        ax.text(col_x[1], y, val,   fontsize=6.2, color=_DARK,
                transform=ax.transAxes, va="top", family="monospace")
        ax.text(col_x[2], y, src,   fontsize=6.0, color="#777777",
                transform=ax.transAxes, va="top")
        y -= step


def _draw_returns(ax: plt.Axes, df: pd.DataFrame, sigma: float, ticker: str) -> None:
    rets  = np.diff(np.log(df["adj_close"].values))
    sig_d = sigma / math.sqrt(250)

    ax.plot(rets, color=_BLUE, lw=0.45, alpha=0.9, zorder=2)
    ax.axhline(0,      color=_DARK, lw=0.4)
    ax.axhline( sig_d, color=_RED, lw=0.9, ls="--", label=f"+1σ_d ({sig_d*100:.2f}%)")
    ax.axhline(-sig_d, color=_RED, lw=0.9, ls="--", label=f"−1σ_d")
    ax.set_title(f"{ticker}  —  Daily Log-Returns", fontsize=7, color=_DARK, pad=3)
    ax.set_ylabel("Log-return", fontsize=6)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.8)
    _style_chart(ax)


def _draw_payoff(ax: plt.Axes, result: dict, ticker: str) -> None:
    payoffs = np.asarray(result["payoffs"])
    weights = np.asarray(result["weights"])
    price   = result["price"]

    hist_w, edges = np.histogram(payoffs, bins=150, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw      = (edges[1] - edges[0]) * 0.9
    colors  = np.where(centers > 0, _BLUE, _MID)

    ax.bar(centers, hist_w, color=colors, width=bw, alpha=0.85, zorder=2)
    ax.axvline(price, color=_GOLD, lw=2.0, ls="--",
               label=f"Price = ${price:.2f}", zorder=3)
    ax.set_title(f"{ticker}  —  Terminal Payoff Distribution", fontsize=7, color=_DARK, pad=3)
    ax.set_xlabel("Payoff ($)", fontsize=6)
    ax.set_ylabel("Risk-neutral probability", fontsize=6)
    ax.legend(fontsize=5.5)
    _style_chart(ax)


def _draw_convergence(
    ax: plt.Axes, conv: pd.DataFrame, price: float, n_ref: int, ticker: str
) -> None:
    ax.plot(conv["n"], conv["price"], color=_BLUE, lw=1.2,
            marker="o", ms=3, label="Price(n)", zorder=2)
    ax.axhline(price, color=_GOLD, lw=1.0, ls="--",
               label=f"n={n_ref}: ${price:.3f}", zorder=3)
    ax.set_title(f"{ticker}  —  Step Convergence", fontsize=7, color=_DARK, pad=3)
    ax.set_xlabel("Steps (n)", fontsize=6)
    ax.set_ylabel("Option price ($)", fontsize=6)
    ax.legend(fontsize=5.5)
    _style_chart(ax)


def _draw_methodology(ax: plt.Axes, params: ModelParams) -> None:
    ax.axis("off")
    sig_d  = params.sigma / math.sqrt(250)
    e_rdt  = math.exp(params.r * params.dt)

    body = (
        "Instrument\n"
        "  European floating-strike Asian call. Payoff at maturity T:\n"
        "    Π = max( Sₙ − Ā , 0 )\n"
        "    Ā = arithmetic mean of all Sₜ along the price path\n"
        "\n"
        "Pricing — CRR Binomial Tree\n"
        "  u = exp(σ √Δt)      d = 1/u      Δt = T/n\n"
        "  q = (e^{rΔt} − d) / (u − d)    [risk-neutral prob.]\n"
        "  Price = e^{−rT} × E^Q[ Π ]\n"
        "  Path merging: states with equal (price, cum_sum)\n"
        "  are collapsed, reducing complexity significantly.\n"
        "\n"
        "Volatility assumption\n"
        "  Backward-looking: daily log-returns computed over the\n"
        "  full historical window (2020–present), std with ddof=1,\n"
        f"  annualised × √250 (M=250 trading days/year, assignment spec).\n"
        f"  σ_daily = {sig_d * 100:.3f}%  →  σ_annual = {params.sigma * 100:.2f}%\n"
        "\n"
        "Physical vs risk-neutral probability\n"
        "  p = 0.5 (assigned, real-world up-move probability).\n"
        "  Not used in pricing — included for reference only.\n"
        "  Pricing is performed exclusively under Q (risk-neutral).\n"
        "\n"
        "Risk-free rate\n"
        f"  r = {params.r * 100:.2f}% p.a. (annual continuous, input).\n"
        f"  Per-step growth factor: e^{{rΔt}} = {e_rdt:.6f}"
    )

    ax.text(0.02, 0.99, body, fontsize=5.9, color=_DARK,
            transform=ax.transAxes, va="top", linespacing=1.55)
    ax.text(0.02, 1.04, "METHODOLOGY", fontsize=7, fontweight="bold",
            color=_BLUE, transform=ax.transAxes, va="bottom")


def _draw_footer(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.plot([0, 1], [1.0, 1.0], color=_BLUE, lw=1.5, transform=ax.transAxes)
    note = (
        "For academic and research purposes only. "
        "Prices computed via CRR binomial tree on adjusted-close prices sourced from Yahoo Finance. "
        "Volatility is backward-looking and does not constitute investment advice."
    )
    ax.text(0.0, 0.75, note, fontsize=5.0, color="#888888",
            transform=ax.transAxes, va="top")


# ── Public API ────────────────────────────────────────────────────

def generate_report(
    params: ModelParams,
    result: dict,
    df: pd.DataFrame,
    approx: float,
    convergence: pd.DataFrame | None = None,
    sigma_window: str = "full",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render a one-page overnight research report (A4 portrait, 150 dpi).

    Parameters
    ----------
    params       : calibrated ModelParams
    result       : dict returned by price_asian_call()
    df           : historical price DataFrame with 'adj_close' column
    approx       : normal-approximation price
    convergence  : DataFrame from step_convergence() — fills bottom-left panel
    sigma_window : label for the volatility estimation window shown in the table
    save_path    : file path to save (.pdf recommended, .png also works)

    Returns
    -------
    matplotlib Figure (caller can plt.show() or close it)
    """
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=_WHITE)
    gs  = gridspec.GridSpec(
        5, 2, figure=fig,
        height_ratios=[0.07, 0.22, 0.26, 0.32, 0.07],
        hspace=0.55, wspace=0.28,
        left=0.06, right=0.96, top=0.96, bottom=0.04,
    )

    _draw_header(      fig.add_subplot(gs[0, :]),   params.ticker)
    _draw_price_box(   fig.add_subplot(gs[1, 0]),   result["price"], approx, result["n_states"])
    _draw_params_table(fig.add_subplot(gs[1, 1]),   params, sigma_window)
    _draw_returns(     fig.add_subplot(gs[2, 0]),   df, params.sigma, params.ticker)
    _draw_payoff(      fig.add_subplot(gs[2, 1]),   result, params.ticker)

    ax_conv = fig.add_subplot(gs[3, 0])
    if convergence is not None:
        _draw_convergence(ax_conv, convergence, result["price"], params.n, params.ticker)
    else:
        ax_conv.axis("off")
        ax_conv.text(
            0.5, 0.5,
            "Convergence chart\n(pass show_convergence=True\nto populate this panel)",
            ha="center", va="center", fontsize=7, color="#aaaaaa",
            transform=ax_conv.transAxes,
        )

    _draw_methodology(fig.add_subplot(gs[3, 1]), params)
    _draw_footer(      fig.add_subplot(gs[4, :]))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_WHITE)
        print(f"[report] saved -> {save_path}")

    return fig
