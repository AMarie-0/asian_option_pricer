# data/database.py
"""
Persist and retrieve price data using DuckDB.

Exports
-------
save_prices   – write a price DataFrame to DuckDB
load_prices   – read prices back as a DataFrame
db_info       – print a summary of what is stored
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


# ── Default DB path ───────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.duckdb"


# ── 1. Save ───────────────────────────────────────────────────────────────────

def save_prices(
    df: pd.DataFrame,
    ticker: str,
    db_path: Path = DB_PATH,
) -> None:
    """
    Write `df` to the `prices` table in DuckDB.
    If rows for this ticker already exist they are replaced.

    Parameters
    ----------
    df       : DataFrame with columns [date, adj_close, ticker].
    ticker   : e.g. 'AAPL'
    db_path  : path to the .duckdb file (created if absent).
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # tag every row with the ticker so one table holds multiple assets
    df = df.copy()
    df["ticker"] = ticker.upper()

    with duckdb.connect(str(db_path)) as con:

        # create table if it does not exist yet
        con.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                date      DATE        NOT NULL,
                adj_close DOUBLE      NOT NULL,
                ticker    VARCHAR(10) NOT NULL,
                PRIMARY KEY (date, ticker)
            )
        """)

        # delete stale rows for this ticker then re-insert
        con.execute("DELETE FROM prices WHERE ticker = ?", [ticker.upper()])
        con.execute("INSERT INTO prices SELECT date, adj_close, ticker FROM df")

    print(f"[database] saved {len(df)} rows for {ticker.upper()} → {db_path}")


# ── 2. Load ───────────────────────────────────────────────────────────────────

def load_prices(
    ticker: str,
    start: str | None = None,
    end:   str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Read prices for `ticker` from DuckDB.

    Parameters
    ----------
    ticker    : e.g. 'AAPL'
    start/end : optional ISO date strings '2020-01-01' to filter the range.
    db_path   : path to the .duckdb file.

    Returns
    -------
    DataFrame with columns [date, adj_close], sorted by date.
    """

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run fetch.py first."
        )

    query = "SELECT date, adj_close FROM prices WHERE ticker = ?"
    params: list = [ticker.upper()]

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY date ASC"

    with duckdb.connect(str(db_path)) as con:
        df = con.execute(query, params).df()

    if df.empty:
        raise ValueError(
            f"No data found for {ticker.upper()} in the given date range."
        )

    df["date"] = pd.to_datetime(df["date"])
    print(f"[database] loaded {len(df)} rows for {ticker.upper()}")
    return df


# ── 3. Info ───────────────────────────────────────────────────────────────────

def db_info(db_path: Path = DB_PATH) -> None:
    """Print a summary of everything stored in the database."""

    if not db_path.exists():
        print(f"[database] no database found at {db_path}")
        return

    with duckdb.connect(str(db_path)) as con:
        summary = con.execute("""
            SELECT
                ticker,
                COUNT(*)        AS rows,
                MIN(date)       AS first_date,
                MAX(date)       AS last_date,
                MIN(adj_close)  AS price_min,
                MAX(adj_close)  AS price_max
            FROM prices
            GROUP BY ticker
            ORDER BY ticker
        """).df()

    print("\n── Database Summary ─────────────────────────────────")
    print(summary.to_string(index=False))
    print("─────────────────────────────────────────────────────\n")
