# data/fetch.py

import yfinance as yf
import pandas as pd
from datetime import datetime, date

# ── constants ────────────────────────────────────────────────────────────────

SUPPORTED_TICKERS = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "TSLA":  "Tesla Inc.",
    "NVDA":  "NVIDIA Corp.",
    "JPM":   "JPMorgan Chase",
    "GS":    "Goldman Sachs",
    "JNJ":   "Johnson & Johnson",
    "XOM":   "ExxonMobil",
    "SPY":   "S&P 500 ETF",
}

RISK_FREE_TICKER = "^IRX"   # 3-month T-bill annualised yield (%)

MIN_YEARS_WARN = 1
MAX_YEARS_WARN = 15

# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker not in SUPPORTED_TICKERS:
        supported = ", ".join(SUPPORTED_TICKERS.keys())
        raise ValueError(
            f"'{ticker}' is not supported.\n"
            f"Supported tickers: {supported}"
        )
    return ticker


def _validate_dates(start: date, end: date) -> None:
    if start >= end:
        raise ValueError("start_date must be before end_date.")

    years = (end - start).days / 365.25

    if years < MIN_YEARS_WARN:
        print(
            f"[WARNING] Date range is only {years:.1f} year(s). "
            "Short periods may produce unreliable volatility estimates."
        )
    if years > MAX_YEARS_WARN:
        print(
            f"[WARNING] Date range spans {years:.1f} years. "
            "Very long periods may mix different volatility regimes "
            "and reduce the relevance of the estimate."
        )


def _parse_date(d: str | date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    # string — try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{d}'. Use 'YYYY-MM-DD', 'DD/MM/YYYY', or 'MM/DD/YYYY'."
    )

# ── main fetch functions ───────────────────────────────────────────────────────

def fetch_prices(
    ticker: str,
    start_date: str | date | datetime = "2020-01-01",
    end_date:   str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a supported equity ticker.

    Parameters
    ----------
    ticker     : e.g. 'AAPL'
    start_date : first date to include (default '2020-01-01')
    end_date   : last date to include  (default: today)

    Returns
    -------
    DataFrame with columns ['date', 'adj_close', 'ticker']
    sorted ascending by date, no missing values.
    """
    ticker = _validate_ticker(ticker)

    start = _parse_date(start_date)
    end   = _parse_date(end_date) if end_date else date.today()

    _validate_dates(start, end)

    print(f"[fetch] Downloading {ticker} ({SUPPORTED_TICKERS[ticker]}) "
          f"from {start} to {end} ...")

    raw = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=True,   # gives us adjusted close directly
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check your date range or internet connection."
        )

    # yfinance returns MultiIndex columns when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = (
        raw[["Close"]]
        .rename(columns={"Close": "adj_close"})
        .reset_index()
        .rename(columns={"Date": "date"})
    )

    df["date"]   = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker

    df = df.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)

    print(f"[fetch] {len(df)} observations retrieved for {ticker}.")
    return df[["date", "ticker", "adj_close"]]


def fetch_risk_free_rate() -> float:
    """
    Fetch the current 3-month T-bill yield from Yahoo Finance (^IRX).
    Returns the annualised rate as a decimal (e.g. 0.0523 for 5.23%).
    Falls back to 0.01 if the fetch fails.
    """
    try:
        raw = yf.download(RISK_FREE_TICKER, period="5d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        latest_yield = float(raw["Close"].dropna().iloc[-1])
        rate = latest_yield / 100
        print(f"[fetch] Risk-free rate (^IRX): {latest_yield:.3f}% → r = {rate:.4f}")
        return rate
    except Exception as e:
        print(f"[WARNING] Could not fetch risk-free rate ({e}). Defaulting to r = 0.01.")
        return 0.01
