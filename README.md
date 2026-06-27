<!-- README.md -->
# Asian Option Pricer 🌏📈

Price a **European floating-strike Asian call option** on any supported
equity using a **multi-period binomial tree**, complete with:

- ✅ Automatic data download & DuckDB caching
- ✅ CRR calibration from historical volatility
- ✅ Full path enumeration (exact Asian pricing)
- ✅ Normal approximation benchmark
- ✅ Robustness checks across volatility windows
- ✅ Step-convergence analysis
- ✅ Interactive 3D price surface (Plotly)
- ✅ Binomial tree visualisation

## Quick Start

```bash
git clone https://github.com/yourname/asian-option-pricer
cd asian-option-pricer
pip install -e ".[dev]"
