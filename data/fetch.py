# data/fetch.py

import yfinance as yf
import pandas as pd
from datetime import datetime, date

# ── SSL bypass ────────────────────────────────────────────────────────────────
# yfinance ≥1.0 uses curl_cffi as its HTTP backend; on machines where the
# system CA bundle doesn't include a corporate proxy cert, every download
# fails with curl error 60 (SSL certificate problem).  Passing verify=False
# to the underlying curl_cffi Session disables peer verification.
try:
    from curl_cffi import requests as _curl_requests
    _YF_SESSION = _curl_requests.Session(verify=False)
except ImportError:
    _YF_SESSION = None   # fall back to yfinance default (no session override)

# ── constants ────────────────────────────────────────────────────────────────
# Reference list of well-known tickers — any valid Yahoo Finance symbol works.
# Full symbol directory: https://finance.yahoo.com/lookup/

COMMON_TICKERS = {
    # US large-cap equities
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "TSLA":  "Tesla Inc.",
    "NVDA":  "NVIDIA Corp.",
    "BRK-B": "Berkshire Hathaway B",
    "JPM":   "JPMorgan Chase",
    "GS":    "Goldman Sachs",
    "JNJ":   "Johnson & Johnson",
    "XOM":   "ExxonMobil",
    "V":     "Visa Inc.",
    "MA":    "Mastercard Inc.",
    "UNH":   "UnitedHealth Group",
    "BAC":   "Bank of America",
    "WMT":   "Walmart Inc.",
    "PG":    "Procter & Gamble",
    "HD":    "Home Depot Inc.",
    "CVX":   "Chevron Corp.",
    # ETFs
    "SPY":   "S&P 500 ETF (SPDR)",
    "QQQ":   "Nasdaq-100 ETF (Invesco)",
    "DIA":   "Dow Jones ETF (SPDR)",
    "IWM":   "Russell 2000 ETF (iShares)",
    "GLD":   "Gold ETF (SPDR)",
    "TLT":   "20+ Year Treasury ETF (iShares)",
    # International
    "TSM":   "Taiwan Semiconductor",
    "BABA":  "Alibaba Group",
    "ASML":  "ASML Holding",
    "SAP":   "SAP SE",
    "TM":    "Toyota Motor",
    "SHEL":  "Shell plc",
}

RISK_FREE_TICKER = "^IRX"   # 3-month T-bill annualised yield (%)

MIN_YEARS_WARN = 1
MAX_YEARS_WARN = 15

# ── helpers ───────────────────────────────────────────────────────────────────


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
    ticker = ticker.upper().strip()

    start = _parse_date(start_date)
    end   = _parse_date(end_date) if end_date else date.today()

    _validate_dates(start, end)

    name = COMMON_TICKERS.get(ticker, "unknown")
    print(f"[fetch] Downloading {ticker} ({name}) from {start} to {end} ...")

    raw = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=True,   # gives us adjusted close directly
        progress=False,
        **( {"session": _YF_SESSION} if _YF_SESSION is not None else {} ),
    )

    if raw.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Possible causes: invalid ticker symbol, date range outside trading history, "
            "or SSL/network error. Verify the symbol at https://finance.yahoo.com/lookup/"
        )

    # yfinance returns MultiIndex columns when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].rename(columns={"Close": "adj_close"}).reset_index()
    df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "index"))
    df = df.rename(columns={date_col: "date"})


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
        raw = yf.download(RISK_FREE_TICKER, period="5d", progress=False, auto_adjust=True,
                          **( {"session": _YF_SESSION} if _YF_SESSION is not None else {} ))
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        latest_yield = float(raw["Close"].dropna().iloc[-1])
        rate = latest_yield / 100
        print(f"[fetch] Risk-free rate (^IRX): {latest_yield:.3f}% → r = {rate:.4f}")
        return rate
    except Exception as e:
        print(f"[WARNING] Could not fetch risk-free rate ({e}). Defaulting to r = 0.01.")
        return 0.01
    
def fetch_and_cache(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    db_path: str = "data/prices.duckdb",
    force_refresh: bool = False,
) -> "pd.DataFrame":
    """
    Return prices for `ticker`, loading from DuckDB cache when available.
    Downloads fresh data and saves it when the cache is empty or force_refresh=True.
    """
    from pathlib import Path
    from data.database import save_prices, load_prices

    db = Path(db_path)
    if not force_refresh and db.exists():
        try:
            return load_prices(ticker, start=start, end=end, db_path=db)
        except (FileNotFoundError, ValueError):
            pass

    df = fetch_prices(ticker, start_date=start, end_date=end)
    save_prices(df, ticker, db_path=db)
    return df


if __name__ == "__main__":
    df = fetch_prices("GS", "2020-01-01", "2026-04-28")
    print(df.head(10))
    print(df.tail(5))
    print(f"\nShape: {df.shape}")
    
    r = fetch_risk_free_rate()
    print(f"\nRisk-free rate: {r}")
