import pytest
from data.fetch import fetch_prices, COMMON_TICKERS
from src.model.calibration import calibrate

TICKERS_TO_TEST = list(COMMON_TICKERS.keys())


@pytest.mark.parametrize("ticker", TICKERS_TO_TEST)
def test_calibration(ticker):
    df = fetch_prices(ticker, "2020-01-01", "2026-04-28")
    params = calibrate(df, ticker=ticker, r=0.01, n=25, T=0.5)

    assert 0.05 < params.sigma < 1.0,            f"{ticker}: sigma out of range: {params.sigma}"
    assert abs(params.u * params.d - 1) < 1e-10, f"{ticker}: u*d != 1"
    assert 0 < params.q < 1,                     f"{ticker}: q not a valid probability: {params.q}"
    assert params.S0 > 0,                        f"{ticker}: S0 must be positive: {params.S0}"
    assert params.ticker == ticker
