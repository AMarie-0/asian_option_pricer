# tests/test_pricer.py
from src.model.calibration import ModelParams
from src.model.pricer import price_asian_call
import numpy as np

def _make_params(**kwargs):
    base = dict(ticker="T", S0=100, sigma=0.30, r=0.0002,
                T=0.5, n=10, u=None, d=None, q=None, dt=None)
    base.update(kwargs)
    dt  = base["T"] / base["n"]
    u   = np.exp(base["sigma"] * np.sqrt(dt))
    d   = 1.0 / u
    r_p = (1 + base["r"]) ** (252 * dt) - 1
    q   = (1 + r_p - d) / (u - d)
    base.update(u=u, d=d, q=q, dt=dt)
    return ModelParams(**base)

def test_price_positive():
    p = _make_params()
    r = price_asian_call(p)
    assert r["price"] >= 0

def test_weights_sum_to_one():
    p = _make_params()
    r = price_asian_call(p)
    assert abs(sum(r["weights"]) - 1.0) < 1e-8
