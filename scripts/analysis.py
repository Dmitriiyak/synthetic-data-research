"""
Analysis of experimental results.

Generates:
    figures/
        accuracy_vs_synthetic_ratio.png
        domain_gap.png

Reads:
    results/summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(RESULTS_DIR / "summary.csv")


# ==========================================================
# DATA
# ==========================================================

x = summary["synthetic_train_pct"]

real_acc = summary["real_test_accuracy"]

synthetic_acc = summary["synthetic_test_accuracy"]

domain_gap = synthetic_acc - real_acc


# ==========================================================
# FIGURE 1
# Accuracy
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    real_acc,
    marker="o",
    linewidth=2,
    label="Real Test"
)

plt.plot(
    x,
    synthetic_acc,
    marker="s",
    linewidth=2,
    label="Synthetic Test"
)

plt.title("Accuracy vs Synthetic Training Ratio")

plt.xlabel("Synthetic Images in Training Set (%)")

plt.ylabel("Accuracy (%)")

plt.xticks(x)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "accuracy_vs_synthetic_ratio.png",
    dpi=300
)

plt.show()


# ==========================================================
# FIGURE 2
# Domain Gap
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    domain_gap,
    marker="o",
    linewidth=2
)

plt.title("Domain Gap")

plt.xlabel("Synthetic Images in Training Set (%)")

plt.ylabel("Synthetic Accuracy − Real Accuracy (%)")

plt.xticks(x)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "domain_gap.png",
    dpi=300
)

plt.show()


# ==========================================================
# SUMMARY
# ==========================================================

report = summary.copy()

report["domain_gap"] = (
    report["synthetic_test_accuracy"]
    - report["real_test_accuracy"]
).round(2)

print("\n================ SUMMARY ================\n")

print(
    report[
        [
            "experiment",
            "real_train_pct",
            "synthetic_train_pct",
            "real_test_accuracy",
            "synthetic_test_accuracy",
            "domain_gap",
        ]
    ]
)

print("\n=========================================\n")