# tests/test_calibration.py
from data.fetch import fetch_prices
from data.calibrate import calibrate

df = fetch_prices("AAPL", "2020-01-01", "2026-04-28")

params = calibrate(df, r=0.01, n=25, T=0.5)

params.display()

# sanity checks  (dot access, not dict access)
assert 0.20 < params.sigma < 0.60,          f"sigma out of range: {params.sigma}"
assert abs(params.u * params.d - 1) < 1e-10, "u*d != 1"
assert 0 < params.q < 1,                    f"q not a valid probability: {params.q}"
assert params.S0 > 0,                       f"S0 must be positive: {params.S0}"

print("All checks passed.")
