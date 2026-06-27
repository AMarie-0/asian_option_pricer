# AsianOptionAnalyzer.py
"""
AsianOptionAnalyzer
===================
Main entry point for the Asian option pricing toolkit.

Usage
-----
>>> from AsianOptionAnalyzer import AsianOptionAnalyzer
>>> az = AsianOptionAnalyzer("AAPL", r=0.0002, T=0.5, n=25)
>>> az.run(
...     show_tree=True,
...     show_surface=True,
...     show_robustness=True,
...     show_convergence=True,
...     verbose=True,
... )
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from data.fetch       import fetch_and_cache, SUPPORTED_TICKERS
from data.database    import StockDatabase
from src.model.calibration    import calibrate, ModelParams
from src.model.pricer         import price_asian_call
from src.analytics.robustness import (
    volatility_window_sensitivity,
    step_convergence,
)
from src.analytics.approximation import normal_approximation_price
from src.visualization.tree_plot    import plot_binomial_tree
from src.visualization.payoff_plot  import plot_payoff_distribution
from src.visualization.returns_plot import plot_log_returns
from src.visualization.robustness_plot import (
    plot_window_sensitivity,
    plot_convergence,
)
from src.visualization.surface_plot import plot_price_surface
import pandas as pd
import matplotlib.pyplot as plt


class AsianOptionAnalyzer:
    """
    Prices a European floating-strike Asian call option
    using a multi-period binomial tree.

    Parameters
    ----------
    ticker : str
        Must be in SUPPORTED_TICKERS or already in the local DB.
    r : float
        Daily risk-free rate (default 0.02% ≈ 5% p.a.).
    T : float
        Time to maturity in years (default 0.5).
    n : int
        Number of binomial steps (default 25).
    start, end : str
        Historical data window for volatility estimation.
    force_refresh : bool
        Re-download data even if cached.
    """

    SUPPORTED = SUPPORTED_TICKERS

    def __init__(
        self,
        ticker:        str   = "AAPL",
        r:             float = 0.0002,
        T:             float = 0.5,
        n:             int   = 25,
        start:         str   = "2020-01-01",
        end:           str   = "2026-04-30",
        force_refresh: bool  = False,
        db_path:       str   = "data/stocks.db",
    ):
        ticker = ticker.upper()
        if ticker not in self.SUPPORTED:
            avail = ", ".join(self.SUPPORTED)
            raise ValueError(
                f"'{ticker}' not supported. Choose from: {avail}"
            )

        self.ticker = ticker
        self.r      = r
        self.T      = T
        self.n      = n

        # ── Load data ────────────────────────────────────────────
        self.df: pd.DataFrame = fetch_and_cache(
            ticker, start=start, end=end,
            db_path=db_path, force_refresh=force_refresh
        )

        # ── Calibrate model ──────────────────────────────────────
        self.params: ModelParams = calibrate(
            self.df, ticker, r=r, T=T, n=n
        )

        # ── Pricing results (lazy) ───────────────────────────────
        self._result:      dict | None = None
        self._robustness:  pd.DataFrame | None = None
        self._convergence: pd.DataFrame | None = None

    # ── Properties ───────────────────────────────────────────────

    @property
    def result(self) -> dict:
        if self._result is None:
            self._result = price_asian_call(self.params)
        return self._result

    @property
    def price(self) -> float:
        return self.result["price"]

    # ── Public API ───────────────────────────────────────────────

    def summary(self, verbose: bool = True) -> dict:
        """Print calibration summary and option price."""
        approx = normal_approximation_price(self.params)

        out = {
            "ticker":            self.ticker,
            "S0":                self.params.S0,
            "sigma":             self.params.sigma,
            "r":                 self.r,
            "T":                 self.T,
            "n":                 self.n,
            "u":                 self.params.u,
            "d":                 self.params.d,
            "q":                 self.params.q,
            "option_price":      self.price,
            "normal_approx":     approx,
            "n_terminal_states": self.result["n_states"],
        }

        if verbose:
            print(self.params.summary())
            print(f"\n  Option Price (binomial) : ${self.price:.4f}")
            print(f"  Option Price (approx)   : ${approx:.4f}")
            print(f"  # terminal states       : {self.result['n_states']:,}")
        return out

    def run(
        self,
        show_returns:     bool = True,
        show_tree:        bool = False,
        show_payoff:      bool = True,
        show_robustness:  bool = True,
        show_convergence: bool = True,
        show_surface:     bool = False,
        tree_max_steps:   int  = 8,
        verbose:          bool = True,
        save_figures:     bool = False,
        output_dir:       str  = "outputs/",
    ) -> dict:
        """
        Run the full analysis pipeline.

        Parameters
        ----------
        show_returns     : plot log-returns with ±1σ bands
        show_tree        : draw the binomial tree diagram
        show_payoff      : plot payoff distribution at terminal nodes
        show_robustness  : volatility-window sensitivity table & chart
        show_convergence : price vs n-steps convergence chart
        show_surface     : 3D (σ, r) → price surface (Plotly)
        tree_max_steps   : how many steps to render in the tree plot
        verbose          : print calibration & pricing summary
        save_figures     : save all figures to output_dir
        """
        import os
        if save_figures:
            os.makedirs(output_dir, exist_ok=True)

        # ── 1. Summary ────────────────────────────────────────────
        results = self.summary(verbose=verbose)

        # ── 2. Log returns ────────────────────────────────────────
        if show_returns:
            fig = plot_log_returns(self.df, self.ticker, self.params.sigma)
            if save_figures:
                fig.savefig(f"{output_dir}{self.ticker}_returns.png", dpi=150)
            plt.show()

        # ── 3. Binomial tree ──────────────────────────────────────
        if show_tree:
            fig = plot_binomial_tree(
                self.params, max_steps=tree_max_steps,
                save_path=f"{output_dir}{self.ticker}_tree.png"
                          if save_figures else None
            )
            plt.show()

        # ── 4. Payoff distribution ────────────────────────────────
        if show_payoff:
            fig = plot_payoff_distribution(
                self.result, self.params,
                save_path=f"{output_dir}{self.ticker}_payoff.png"
                          if save_figures else None
            )
            plt.show()

        # ── 5. Robustness ─────────────────────────────────────────
        if show_robustness:
            self._robustness = volatility_window_sensitivity(
                self.df, self.ticker, r=self.r, T=self.T, n=self.n
            )
            if verbose:
                print("\n── Robustness (Volatility Window) ──")
                print(self._robustness.to_string(index=False))
            fig = plot_window_sensitivity(
                self._robustness, self.ticker,
                save_path=f"{output_dir}{self.ticker}_robustness.png"
                          if save_figures else None
            )
            plt.show()
            results["robustness"] = self._robustness

        # ── 6. Convergence ────────────────────────────────────────
        if show_convergence:
            self._convergence = step_convergence(
                self.df, self.ticker, r=self.r, T=self.T
            )
            if verbose:
                print("\n── Convergence (Steps) ──")
                print(self._convergence.to_string(index=False))
            fig = plot_convergence(
                self._convergence, self.ticker,
                save_path=f"{output_dir}{self.ticker}_convergence.png"
                          if save_figures else None
            )
            plt.show()
            results["convergence"] = self._convergence

        # ── 7. 3-D Price Surface ──────────────────────────────────
        if show_surface:
            fig = plot_price_surface(self.df, self.ticker, n=self.n)
            if save_figures:
                fig.write_html(f"{output_dir}{self.ticker}_surface.html")
            fig.show()

        return results

    # ── Utilities ─────────────────────────────────────────────────

    @classmethod
    def list_tickers(cls) -> None:
        print("Supported tickers:")
        for k, v in cls.SUPPORTED.items():
            print(f"  {k:<8} {v}")

    @classmethod
    def compare(
        cls,
        tickers: list[str],
        r: float = 0.0002,
        T: float = 0.5,
        n: int   = 25,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Price the Asian call across multiple tickers and return
        a comparison DataFrame.
        """
        rows = []
        for t in tickers:
            az  = cls(t, r=r, T=T, n=n)
            row = az.summary(verbose=False)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("ticker")
        if verbose:
            print(df[["S0","sigma","u","d","q",
                       "option_price","normal_approx"]].to_string())
        return df
