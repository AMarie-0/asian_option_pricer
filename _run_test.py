import sys
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plt.show() is a no-op

from AsianOptionAnalyzer import AsianOptionAnalyzer

az = AsianOptionAnalyzer("AAPL", r=0.0002, T=0.5, n=25)
result = az.run(
    show_returns=True,
    show_tree=True,
    show_payoff=True,
    show_robustness=True,
    show_convergence=True,
    show_surface=False,
    tree_max_steps=8,
    verbose=True,
    save_figures=True,
    output_dir="outputs/",
)

print("\nDONE - figures saved to outputs/")
sys.stdout.flush()
