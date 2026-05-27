# test_calibrate.py
from data.calibrate import calibrate

params = calibrate()

print(f"sigma : {params['sigma']:.6f}")
print(f"u     : {params['u']:.6f}")
print(f"d     : {params['d']:.6f}")
print(f"q     : {params['q']:.6f}")
print(f"dt    : {params['dt']:.6f}")
print(f"S0    : {params['S0']:.2f}")

# sanity checks
assert 0.20 < params['sigma'] < 0.60,  f"sigma out of range: {params['sigma']}"
assert abs(params['u'] * params['d'] - 1) < 1e-10, "u*d != 1"
assert 0 < params['q'] < 1,            f"q not a valid probability: {params['q']}"
assert params['S0'] > 0,               f"S0 must be positive: {params['S0']}"

print("\nAll checks passed.")
