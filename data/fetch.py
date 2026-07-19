# data/fetch.py

import yfinance as yf
import pandas as pd
from datetime import datetime, date

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

RISK_FREE_TICKER = "^IRX"


def _parse_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{d}'.")


def fetch_prices(
    ticker: str,
    start_date="2020-01-01",
    end_date=None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for any Yahoo Finance ticker.
    Returns DataFrame with columns ['date', 'adj_close', 'ticker'].
    """
    ticker = ticker.upper().strip()
    start  = _parse_date(start_date)
    end    = _parse_date(end_date) if end_date else date.today()

    if start >= end:
        raise ValueError("start_date must be before end_date.")

    name = SUPPORTED_TICKERS.get(ticker, ticker)
    print(f"[fetch] Downloading {ticker} ({name}) from {start} to {end} ...")

    raw = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'. "
            "For non-US stocks use the exchange suffix — e.g. SIE.DE (Siemens), "
            "ASML.AS (ASML), BNP.PA (BNP Paribas), AIR.PA (Airbus). "
            "Verify the exact symbol at https://finance.yahoo.com/lookup"
        )

    # Flatten MultiIndex columns that yfinance sometimes produces
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalise column names to lowercase
    raw.columns = [str(c).lower().strip() for c in raw.columns]

    # Pick the close column (auto_adjust=True gives 'close', not 'adj close')
    close_col = next(
        (c for c in raw.columns if c in ("close", "adj close", "adj_close")),
        None
    )
    if close_col is None:
        raise RuntimeError(
            f"Could not find a price column for '{ticker}'. "
            f"Available columns: {list(raw.columns)}"
        )

    df = raw[[close_col]].rename(columns={close_col: "adj_close"}).reset_index()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Rename date/datetime/index column to 'date'
    date_col = next(
        (c for c in df.columns if c in ("date", "datetime", "price")),
        df.columns[0]
    )
    df = df.rename(columns={date_col: "date"})
    df["date"]      = pd.to_datetime(df["date"]).dt.date
    df["ticker"]    = ticker
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

    df = df.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)

    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} observations for '{ticker}' — not enough to estimate volatility. "
            "Try extending the date range or checking the ticker symbol."
        )

    print(f"[fetch] {len(df)} observations retrieved for {ticker}.")
    return df[["date", "ticker", "adj_close"]]


def fetch_risk_free_rate() -> float:
    """
    Fetch current 3-month T-bill yield from Yahoo Finance (^IRX).
    Returns annualised rate as decimal. Falls back to 0.01 on failure.
    """
    try:
        raw = yf.download(RISK_FREE_TICKER, period="5d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().strip() for c in raw.columns]
        latest = float(raw["close"].dropna().iloc[-1])
        rate   = latest / 100
        print(f"[fetch] Risk-free rate (^IRX): {latest:.3f}% → r = {rate:.4f}")
        return rate
    except Exception as e:
        print(f"[WARNING] Could not fetch risk-free rate ({e}). Defaulting to r = 0.01.")
        return 0.01


def fetch_and_cache(
    ticker: str,
    start: str = "2020-01-01",
    end=None,
    db_path: str = "data/prices.duckdb",
    force_refresh: bool = False,
) -> pd.DataFrame:
    from pathlib import Path
    from data.database import save_prices, load_prices

    db = Path(db_path)
    if not force_refresh and db.exists():
        try:
            return load_prices(ticker, start=start, end=end, db_path=db)
        except Exception:
            pass

    df = fetch_prices(ticker, start_date=start, end_date=end)
    try:
        save_prices(df, ticker, db_path=db)
    except Exception:
        pass
    return df