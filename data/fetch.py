# data/fetch.py

import yfinance as yf
import pandas as pd
import numpy as np
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


def _fetch_via_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Try yfinance download."""
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower().strip() for c in raw.columns]

    close_col = next(
        (c for c in raw.columns if c in ("close", "adj close", "adj_close")), None
    )
    if close_col is None:
        return pd.DataFrame()

    df = raw[[close_col]].rename(columns={close_col: "adj_close"}).reset_index()
    df.columns = [str(c).lower().strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ("date", "datetime", "price")), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    return df.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)


def _fetch_via_requests(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: fetch via direct HTTP request to Yahoo Finance."""
    import requests
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=adjsplits"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data   = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes     = result["indicators"]["adjclose"][0]["adjclose"]
        dates = [datetime.utcfromtimestamp(t).date() for t in timestamps]
        df = pd.DataFrame({"date": dates, "adj_close": closes})
        return df.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_prices(
    ticker: str,
    start_date="2020-01-01",
    end_date=None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for any Yahoo Finance ticker.
    Tries yfinance first, falls back to direct HTTP request.
    Returns DataFrame with columns ['date', 'adj_close', 'ticker'].
    """
    ticker = ticker.upper().strip()
    start  = str(_parse_date(start_date))
    end    = str(_parse_date(end_date) if end_date else date.today())

    name = SUPPORTED_TICKERS.get(ticker, ticker)
    print(f"[fetch] Downloading {ticker} ({name}) from {start} to {end} ...")

    # Try yfinance first
    df = _fetch_via_yfinance(ticker, start, end)

    # Fallback to direct HTTP if yfinance fails
    if df.empty:
        print(f"[fetch] yfinance returned empty, trying direct HTTP for {ticker}...")
        df = _fetch_via_requests(ticker, start, end)

    if df.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'. "
            "For non-US stocks use the exchange suffix — e.g. SIE.DE (Siemens), "
            "ASML.AS (ASML), BNP.PA (BNP Paribas), AIR.PA (Airbus). "
            "Verify the exact symbol at https://finance.yahoo.com/lookup"
        )

    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} observations for '{ticker}' — not enough to estimate volatility."
        )

    df["ticker"] = ticker
    print(f"[fetch] {len(df)} observations retrieved for {ticker}.")
    return df[["date", "ticker", "adj_close"]]


def fetch_risk_free_rate() -> float:
    """
    Fetch current 3-month T-bill yield. Falls back to 0.05 if unavailable.
    """
    try:
        raw = yf.download(RISK_FREE_TICKER, period="5d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().strip() for c in raw.columns]
        latest = float(raw["close"].dropna().iloc[-1])
        return latest / 100
    except Exception:
        pass

    # Fallback via direct HTTP
    try:
        import requests
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX?period1=0&period2=9999999999&interval=1d"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = resp.json()
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        latest = next(c for c in reversed(closes) if c is not None)
        return latest / 100
    except Exception:
        print("[WARNING] Could not fetch risk-free rate. Defaulting to r = 0.05.")
        return 0.05


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