"""SHAP social-feature beeswarms as ONE 2x2 grid sized for a single IEEE column.

Same models, same seed, same SHAP estimators as make_shap_beeswarm.py -- only the
layout changes: four panels in a 2x2 grid drawn at final size (3.45in wide) so no
downscaling happens in LaTeX, with shortened labels and one shared colour bar.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from electricity_forecasting.config import load_config
from electricity_forecasting.features import feature_columns, make_supervised
from electricity_forecasting.models import create_model
from electricity_forecasting.pipeline import load_processed
from electricity_forecasting.sensitivity import (
    lstm_shap_values,
    select_social_feature_columns,
    tree_shap_values,
)
from electricity_forecasting.split import last_year_train_test_split

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "artifacts" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

PRETTY = {"random_forest": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "lstm": "LSTM"}

# Aggressively short labels: a single column leaves ~1.7in per panel, of which the
# tick labels may claim at most about half.
SHORT = {
    "Average Inflation (CPI) - General__2": "Inflation (CPI) 2",
    "Average Inflation (CPI) - General": "Inflation (CPI)",
    "Per Capita Availability of Power": "PC Power Avail.",
    "Installed Capacity of Power": "Installed Capacity",
    "Power Requirement": "Power Req.",
    "State-wise Total Persons Engaged": "Persons Engaged",
    "State-wise Number of Factories": "Factories",
    "State-wise Medium & Small Scale Industries - Total Number of Units": "MSME Units",
    "Electricity Transmission & Distribution Losses": "T&D Losses",
    "Unemployement rate (Adjusted) (Rural)": "Unemp. (Rural)",
    "Unemployement rate (Adjusted) (Urban)": "Unemp. (Urban)",
    "Life Expactancy": "Life Expectancy",
    "Per Capita Net State Domestic Product (Current Prices)": "PC NSDP",
    "Gross State Domestic Product (Current Prices)": "GSDP",
    "Population in Rural Area": "Rural Pop.",
    "Population in Urban Area": "Urban Pop.",
    "Density of Population": "Pop. Density",
    "Poverty Rate": "Poverty Rate",
    "Net Sown Area": "Net Sown Area",
}

MAX_DISPLAY = 8
CACHE = BASE / "artifacts" / "shap_grid_cache.pkl"

config = load_config(BASE / "config" / "default.yaml")
data = load_processed(config)
supervised = make_supervised(data, config.target, config.lags, config.rolling_windows)
train, _ = last_year_train_test_split(supervised)
features = feature_columns(supervised, config.target)
social = select_social_feature_columns(features)
print(f"{len(features)} features, {len(social)} social", flush=True)

INK = "#1b2a33"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 5.6,
    "axes.labelsize": 5.6, "xtick.labelsize": 5.2, "ytick.labelsize": 5.4,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#b8c4c9",
})

if CACHE.exists():
    print("loading cached SHAP values", flush=True)
    results = pickle.loads(CACHE.read_bytes())
else:
    results = {}
    for name in ["random_forest", "xgboost", "lightgbm"]:
        print(f"fitting {name}...", flush=True)
        model = create_model(name, config.seed).fit(train, config.target, features)
        results[name] = tree_shap_values(model, train, config.target, features, seed=config.seed)

    print("fitting lstm (this takes a few minutes)...", flush=True)
    model = create_model("lstm", config.seed).fit(train, config.target, features)
    results["lstm"] = lstm_shap_values(model, train, config.target, seed=config.seed)
    CACHE.write_bytes(pickle.dumps(results))

fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.55))
for ax, name in zip(axes.ravel(), ["random_forest", "xgboost", "lightgbm", "lstm"]):
    sample, shap_values = results[name]
    cols = [c for c in social if c in sample.columns]
    pos = [sample.columns.get_loc(c) for c in cols]
    sub = sample[cols].copy()
    sub.columns = [SHORT.get(c, c) for c in cols]

    plt.sca(ax)
    shap.summary_plot(shap_values[:, pos], sub, max_display=MAX_DISPLAY, show=False,
                      plot_size=None, color_bar=False, sort=True)
    ax = plt.gca()
    unit = "standardized" if name == "lstm" else "MW"
    ax.set_xlabel(f"SHAP value ({unit})", fontsize=5.4, labelpad=1.5)
    ax.set_title(PRETTY[name], fontsize=6.6, weight="bold", color=INK, pad=3)
    ax.tick_params(labelsize=5.2, pad=1)
    ax.tick_params(axis="y", length=0)

fig.tight_layout(h_pad=1.1, w_pad=0.6, rect=(0, 0.045, 1, 1))
cax = fig.add_axes([0.30, 0.005, 0.42, 0.016])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=shap.plots.colors.red_blue),
                  cax=cax, orientation="horizontal")
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.set_label("Feature value", fontsize=5.4, labelpad=1.5, color=INK)
cb.ax.tick_params(labelsize=5.2, length=0, pad=1)
cb.outline.set_visible(False)

path = OUT / "fig_shap_beeswarm_compact.png"
fig.savefig(path)
plt.close(fig)
print("wrote", path, flush=True)


