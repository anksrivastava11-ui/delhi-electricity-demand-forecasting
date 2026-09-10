"""Regenerate the all-model forecast comparison as a 2-column x 5-row grid.

Same data and same panel content as make_paper_figures.py section 6; only the
grid shape and the target size change -- drawn at 3.45in wide so it sits in one
IEEE column at \\linewidth with no downscaling of its text.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from electricity_forecasting.model_comparison import build_period_comparison

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "paper_figures"

INK = "#1b2a33"
MUTED = "#5c6b73"
GRID = "#d9e2e6"
ACCENT = "#0b6e8a"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans", "font.size": 6,
    "axes.labelcolor": INK, "axes.edgecolor": "#b8c4c9", "axes.linewidth": 0.6,
    "axes.facecolor": "white", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.9,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
})

PRETTY = {"seasonal_naive": "Seasonal naive", "holt_winters": "Holt-Winters",
          "arima": "ARIMA", "sarimax": "SARIMAX", "prophet": "Prophet",
          "random_forest": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "lstm": "LSTM"}

order = ["sarimax", "random_forest", "xgboost", "lightgbm", "lstm",
         "holt_winters", "prophet", "seasonal_naive", "arima"]
palette = ["#0b6e8a", "#2a9d8f", "#7bb662", "#e0a458", "#d1495b",
           "#9b5de5", "#f15bb5", "#8d99ae", "#b07d62"]

metrics = pd.read_csv(ART / "model_metrics_test.csv")
raw_cmp, _ = build_period_comparison(metrics, "test")
metric_lookup = raw_cmp.set_index("model")

fig, axes = plt.subplots(5, 2, figsize=(3.45, 3.2), sharex=True, sharey=True)
flat = axes.ravel()
for ax, (name, color) in zip(flat, zip(order, palette)):
    f = pd.read_csv(ART / f"forecast_{name}.csv", parse_dates=["Date"])
    ax.plot(f["Date"], f["actual"], color="#c3ccd1", lw=1.1, zorder=1)
    ax.plot(f["Date"], f["test_prediction"], color=color, lw=0.75, zorder=2)
    r = metric_lookup.loc[name]
    ax.set_title(PRETTY[name], fontsize=6.2, color=INK, pad=2)
    ax.text(0.035, 0.06, f"RMSE {r['RMSE']:,.0f}   MBE {r['MBE']:+,.0f}",
            transform=ax.transAxes, fontsize=4.9, color=MUTED)
    ax.set_ylim(2500, 8100)
    ax.margins(x=0.01)
    ax.tick_params(labelsize=5.2, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# Ninth panel fills the grid; the tenth cell carries the legend instead.
legend_ax = flat[9]
legend_ax.axis("off")
handles = [plt.Line2D([], [], color="#c3ccd1", lw=1.6),
           plt.Line2D([], [], color=ACCENT, lw=1.1)]
legend_ax.legend(handles, ["Actual demand", "Model forecast"], loc="center",
                 fontsize=5.8, frameon=True, framealpha=0.95,
                 edgecolor="#c9d4d9", handlelength=1.6, borderpad=0.7)

for ax in axes[:, 0]:
    ax.set_ylabel("Demand (MW)", fontsize=5.8, labelpad=1.5)
ticks = [pd.Timestamp("2022-10-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-06-01")]
labels = ["Oct '22", "Feb '23", "Jun '23"]
for ax in (axes[4, 0], axes[3, 1]):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.tick_params(axis="x", labelbottom=True)

fig.tight_layout(h_pad=0.55, w_pad=0.5)
path = OUT / "fig_all_model_forecasts_compact.png"
fig.savefig(path)
plt.close(fig)
print("wrote", path, flush=True)


