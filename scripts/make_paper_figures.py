"""Generate publication-quality figures for the IEEE paper.

Reads only existing pipeline artifacts (no model refitting except the univariate
SARIMAX series, which is produced separately) and writes styled figures into
artifacts/paper_figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from electricity_forecasting.model_comparison import build_period_comparison

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "Unrestricted demand"

# ---------------------------------------------------------------- global style
INK = "#1b2a33"
MUTED = "#5c6b73"
GRID = "#d9e2e6"
ACCENT = "#0b6e8a"
WARM = "#d1495b"
GOLD = "#e0a458"
GREEN = "#2a9d8f"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.titlepad": 9,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#b8c4c9",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#c9d4d9",
    "legend.fontsize": 8,
    "text.color": INK,
})


def finish(ax, title=None, xlabel=None, ylabel=None, spines=("top", "right")):
    for side in spines:
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name, flush=True)


PRETTY = {
    "seasonal_naive": "Seasonal naive",
    "holt_winters": "Holt-Winters",
    "arima": "ARIMA",
    "sarimax": "SARIMAX",
    "prophet": "Prophet",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}

# ------------------------------------------------------------------- load data
def load_processed_daily():
    """Read the prepared dataset, in whichever form `prepare` wrote it."""
    stem = BASE / "data" / "processed" / "daily_demand"
    if stem.with_suffix(".parquet").exists():
        frame = pd.read_parquet(stem.with_suffix(".parquet"))
    elif stem.with_suffix(".csv").exists():
        frame = pd.read_csv(stem.with_suffix(".csv"))
    else:
        raise FileNotFoundError(
            f"No prepared dataset at {stem}.parquet or {stem}.csv - run "
            "'electricity-forecast prepare --config config/default.yaml' first."
        )
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


daily = load_processed_daily().sort_values("Date")
metrics = pd.read_csv(ART / "model_metrics_test.csv")
raw_cmp, norm_cmp = build_period_comparison(metrics, "test")
TRAIN_END = pd.Timestamp("2022-07-31")
TEST_START = pd.Timestamp("2022-08-01")

# The Oct-Nov season is stored under the column name "Pre Monsoon" in the source
# workbooks, but it is the post-monsoon season; display it under the correct name.
SEASONS = ["Winter", "Summer", "Monsoon", "Pre Monsoon"]
SEASON_LABELS = {"Winter": "Winter", "Summer": "Summer", "Monsoon": "Monsoon",
                 "Pre Monsoon": "Post Monsoon"}
SEASON_COLORS = {"Winter": "#3d7ea6", "Summer": WARM, "Monsoon": GREEN, "Pre Monsoon": GOLD}


def season_of(row):
    for s in SEASONS:
        if row[s] == 1:
            return s
    return None


daily["Season"] = daily.apply(season_of, axis=1)

# ============================================================ 1. demand series
fig, ax = plt.subplots(figsize=(7.1, 3.0))
ax.axvspan(TEST_START, daily["Date"].max(), color=WARM, alpha=0.09, zorder=0)
ax.plot(daily["Date"], daily[TARGET], color=ACCENT, lw=0.65, alpha=0.85)
ax.plot(daily["Date"], daily[TARGET].rolling(30, center=True).mean(),
        color="#08343f", lw=1.6, label="30-day rolling mean")
ax.axvline(TRAIN_END, color=WARM, lw=1.1, ls="--")
ax.set_ylim(1900, 9700)
ax.annotate("held-out test year\n2022-08-01 to 2023-07-31",
            xy=(TEST_START - pd.Timedelta(days=30), 8950), ha="right", va="center",
            fontsize=7.2, color=WARM, weight="semibold")
ax.legend(loc="upper left")
finish(ax, "Delhi daily unrestricted demand, 2014-2023", None, "Demand (MW)")
save(fig, "fig_daily_demand.png")

# ================================================ 2. season + month structure
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
ax = axes[0]
data = [daily.loc[daily["Season"] == s, TARGET].dropna().values for s in SEASONS]
bp = ax.boxplot(data, tick_labels=[SEASON_LABELS[s].replace(" ", "\n") for s in SEASONS],
                patch_artist=True,
                widths=0.6, medianprops=dict(color="white", lw=1.6),
                flierprops=dict(marker="o", ms=2, mfc=MUTED, mec="none", alpha=0.35),
                whiskerprops=dict(color=MUTED, lw=0.9), capprops=dict(color=MUTED, lw=0.9))
for patch, s in zip(bp["boxes"], SEASONS):
    patch.set_facecolor(SEASON_COLORS[s])
    patch.set_alpha(0.85)
    patch.set_edgecolor("none")
finish(ax, "Demand distribution by season", None, "Demand (MW)")

ax = axes[1]
monthly = daily.groupby(daily["Date"].dt.month)[TARGET].agg(["mean", "std"])
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ax.bar(monthly.index, monthly["mean"], yerr=monthly["std"], color=ACCENT, alpha=0.88,
       width=0.72, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
finish(ax, "Mean demand by calendar month", "Month", "Demand (MW)")
fig.tight_layout()
save(fig, "fig_seasonality.png")

# ==================================================== 3. temperature relation
# Two rows carry a sentinel Max_Temperature of 0 (2016-10-09, 2017-01-29); excluded here.
tmp = daily[daily["Max_Temperature"] > 20]
fig, ax = plt.subplots(figsize=(3.45, 2.85))
sc = ax.scatter(tmp["Max_Temperature"], tmp[TARGET], c=tmp["Date"].dt.dayofyear,
                cmap="twilight_shifted", s=5, alpha=0.6, edgecolors="none")
z = np.polyfit(tmp["Max_Temperature"], tmp[TARGET], 2)
xs = np.linspace(tmp["Max_Temperature"].min(), tmp["Max_Temperature"].max(), 200)
ax.plot(xs, np.polyval(z, xs), color=INK, lw=1.6, label="quadratic fit")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Day of year", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)
ax.legend(loc="upper left")
finish(ax, "Demand vs. maximum temperature", "Max temperature (F)", "Demand (MW)")
save(fig, "fig_temperature.png")

# ======================================================== 4. correlation heatmap
corr_cols = [TARGET, "Max_Temperature", "Min_Temperature", "Summer", "Winter", "Monsoon",
             "Pre Monsoon", "Average Inflation (CPI) - General", "Per Capita Availability of Power",
             "Installed Capacity of Power", "State-wise Number of Factories",
             "State-wise Total Persons Engaged", "Power Requirement", "Life Expactancy",
             "Electricity Transmission & Distribution Losses", "Net Sown Area"]
short = {TARGET: "Demand", "Max_Temperature": "MaxTemp", "Min_Temperature": "MinTemp",
         "Pre Monsoon": "PostMons", "Average Inflation (CPI) - General": "Inflation",
         "Per Capita Availability of Power": "PCPowerAvail",
         "Installed Capacity of Power": "InstCapacity",
         "State-wise Number of Factories": "Factories",
         "State-wise Total Persons Engaged": "PersonsEng",
         "Power Requirement": "PowerReq", "Life Expactancy": "LifeExp",
         "Electricity Transmission & Distribution Losses": "T&D Losses",
         "Net Sown Area": "NetSownArea"}
cm = daily[corr_cols].corr()
labels = [short.get(c, c) for c in corr_cols]
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6.6)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = cm.values[i, j]
        ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                fontsize=5.0, color="white" if abs(v) > 0.55 else INK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)
finish(ax, "Pearson correlation: target, weather, season, socio-economic", spines=())
save(fig, "fig_correlation.png")

# ============================================ 5. SARIMAX 3-line ablation chart
uni = pd.read_csv(ART / "sarimax_univariate_forecast.csv", parse_dates=["Date"])
exo = pd.read_csv(ART / "forecast_sarimax.csv", parse_dates=["Date"])
fig, ax = plt.subplots(figsize=(7.1, 3.1))
ax.plot(uni["Date"], uni["actual"], color=INK, lw=1.5, label="Actual demand", zorder=3)
ax.plot(exo["Date"], exo["test_prediction"], color=ACCENT, lw=1.5,
        label="SARIMAX + annual Fourier exogenous terms (RMSE 514.66, $R^2$ 0.766)", zorder=2)
ax.plot(uni["Date"], uni["test_prediction"], color=WARM, lw=1.5, ls="--",
        label="SARIMAX univariate, no exogenous terms (RMSE 1323.10, $R^2$ -0.548)", zorder=1)
ax.fill_between(uni["Date"], uni["actual"], uni["test_prediction"], color=WARM, alpha=0.10)
lo = min(uni["actual"].min(), uni["test_prediction"].min(), exo["test_prediction"].min())
hi = max(uni["actual"].max(), uni["test_prediction"].max(), exo["test_prediction"].max())
ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
ax.legend(loc="upper center", ncol=1, fontsize=7.2)
ax.margins(x=0.01)
finish(ax, "Effect of exogenous annual Fourier terms on SARIMAX (held-out test year)",
       None, "Demand (MW)")
fig.autofmt_xdate(rotation=0, ha="center")
save(fig, "fig_sarimax_ablation.png")

# ============================================ 6. all-model forecast comparison
order = ["sarimax", "random_forest", "xgboost", "lightgbm", "lstm",
         "holt_winters", "prophet", "seasonal_naive", "arima"]
palette = ["#0b6e8a", "#2a9d8f", "#7bb662", "#e0a458", "#d1495b",
           "#9b5de5", "#f15bb5", "#8d99ae", "#b07d62"]
metric_lookup = raw_cmp.set_index("model")
fig, axes = plt.subplots(3, 3, figsize=(7.1, 5.2), sharex=True, sharey=True)
for ax, (name, color) in zip(axes.ravel(), zip(order, palette)):
    f = pd.read_csv(ART / f"forecast_{name}.csv", parse_dates=["Date"])
    ax.plot(f["Date"], f["actual"], color="#c3ccd1", lw=1.6, zorder=1)
    ax.plot(f["Date"], f["test_prediction"], color=color, lw=1.0, zorder=2)
    r = metric_lookup.loc[name]
    ax.set_title(PRETTY[name], fontsize=9, color=INK, pad=3)
    ax.text(0.03, 0.055, f"RMSE {r['RMSE']:,.0f}   MBE {r['MBE']:+,.0f}",
            transform=ax.transAxes, fontsize=6.6, color=MUTED)
    ax.set_ylim(2500, 8100)
    ax.margins(x=0.01)
    ax.tick_params(labelsize=6.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Demand (MW)", fontsize=8)
for ax in axes[2, :]:
    ax.set_xticks([pd.Timestamp("2022-10-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-06-01")])
    ax.set_xticklabels(["Oct '22", "Feb '23", "Jun '23"], fontsize=6.8)
handles = [plt.Line2D([], [], color="#c3ccd1", lw=2.0),
           plt.Line2D([], [], color=ACCENT, lw=1.4)]
fig.legend(handles, ["Actual demand", "Model forecast"], ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.035), fontsize=8)
fig.suptitle("Test-year forecast of each model against actual demand",
             fontsize=10.5, weight="semibold", color=INK, y=0.995)
fig.tight_layout()
save(fig, "fig_all_model_forecasts.png")

# ------------------------------------------- 6b. recursive error accumulation
recursive_models = ["random_forest", "xgboost", "lightgbm", "lstm", "sarimax"]
rec_colors = {"random_forest": "#2a9d8f", "xgboost": "#7bb662", "lightgbm": GOLD,
              "lstm": WARM, "sarimax": ACCENT}
fig, ax = plt.subplots(figsize=(7.1, 2.7))
for name in recursive_models:
    f = pd.read_csv(ART / f"forecast_{name}.csv", parse_dates=["Date"])
    h = np.arange(1, len(f) + 1)
    err = (f["test_prediction"] - f["actual"]).abs()
    expanding = err.expanding().mean()
    ax.plot(h, expanding, color=rec_colors[name], lw=1.5,
            ls="--" if name == "sarimax" else "-",
            label=PRETTY[name] + (" (non-recursive)" if name == "sarimax" else ""))
ax.set_xlim(1, 365)
ax.legend(ncol=3, loc="lower right", fontsize=7.4)
finish(ax, "Cumulative mean absolute error as the recursive horizon extends",
       "Forecast horizon (days ahead of the forecast origin)", "Cumulative MAE (MW)")
save(fig, "fig_error_accumulation.png")

# ================================================== 7. metric comparison panel
panel = [("RMSE", "RMSE (MW)", False), ("MAPE", "MAPE (%)", False),
         ("R2", "$R^2$", True), ("MBE", "MBE (MW)", None)]
fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
for ax, (metric, label, higher) in zip(axes.ravel(), panel):
    d = raw_cmp.copy()
    if metric == "MBE":
        d["_k"] = d[metric].abs()
        d = d.sort_values("_k")
        colors = [GREEN if v >= 0 else WARM for v in d[metric]]
    else:
        d = d.sort_values(metric, ascending=not higher)
        vals = d[metric].values
        best = vals.max() if higher else vals.min()
        colors = [ACCENT if v == best else "#9fbfcb" for v in vals]
    names = [PRETTY[m] for m in d["model"]]
    bars = ax.barh(names, d[metric], color=colors, height=0.68)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    if metric in ("R2", "MBE"):
        ax.axvline(0, color=MUTED, lw=0.8)
    for b, v in zip(bars, d[metric]):
        off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
        ax.text(b.get_width() + (off if b.get_width() >= 0 else -off), b.get_y() + b.get_height() / 2,
                f"{v:,.0f}" if metric in ("RMSE", "MBE") else f"{v:.2f}",
                va="center", ha="left" if b.get_width() >= 0 else "right", fontsize=6.4, color=MUTED)
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
    finish(ax, label)
fig.tight_layout()
save(fig, "fig_metric_panel.png")

# ==================================================== 8. composite score chart
d = norm_cmp.sort_values("composite_score", ascending=False)
fig, ax = plt.subplots(figsize=(7.1, 2.9))
colors = [ACCENT if i == 0 else "#9fbfcb" for i in range(len(d))]
bars = ax.bar([PRETTY[m] for m in d["model"]], d["composite_score"], color=colors, width=0.66)
for b, v in zip(bars, d["composite_score"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7, color=MUTED)
ax.set_ylim(0, 1.12)
ax.tick_params(axis="x", rotation=18)
for lab in ax.get_xticklabels():
    lab.set_ha("right")
ax.grid(axis="x", visible=False)
finish(ax, "Taxonomy-weighted composite score (error magnitude / bias / fit, 1/3 each)",
       None, "Composite score")
save(fig, "fig_composite.png")
d[["model", "composite_score", "rank"]].to_csv(OUT / "composite_scores.csv", index=False)
print(d[["model", "composite_score", "rank"]].to_string(index=False), flush=True)

# ============================================ 9. SHAP social attribution panel
models4 = ["random_forest", "xgboost", "lightgbm", "lstm"]
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
for ax, m in zip(axes.ravel(), models4):
    s = pd.read_csv(ART / f"shap_values_{m}_social.csv")
    s = s[s["mean_abs_shap"] > 0].sort_values("mean_abs_shap", ascending=False).head(8)
    s = s.iloc[::-1]
    lbl = [t.replace("Average Inflation (CPI) - General__2", "Avg Inflation (CPI) [dup]")
            .replace("Average Inflation (CPI) - General", "Avg Inflation (CPI)")
            .replace("State-wise ", "").replace("Unemployement rate (Adjusted) ", "Unemployment ")
            .replace("Electricity Transmission & Distribution Losses", "Elec. T&D Losses")
            .replace("Life Expactancy", "Life Expectancy")
            .replace("Medium & Small Scale Industries - Total Number of Units", "MSME Units")
           for t in s["feature"]]
    norm = s["mean_abs_shap"] / s["mean_abs_shap"].max()
    ax.barh(lbl, s["mean_abs_shap"], color=plt.cm.YlGnBu(0.35 + 0.5 * norm), height=0.7)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.14)
    unit = "standardized units" if m == "lstm" else "MW"
    finish(ax, PRETTY[m], f"Mean |SHAP| ({unit})")
fig.suptitle("SHAP attribution across the socio-economic covariates, by model",
             fontsize=10.5, weight="semibold", color=INK, y=1.005)
fig.tight_layout()
save(fig, "fig_shap_social.png")

# =================================== 10. overall SHAP (all features), 2 models
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))
for ax, m in zip(axes, ["random_forest", "xgboost"]):
    s = pd.read_csv(ART / f"shap_values_{m}.csv").sort_values("mean_abs_shap", ascending=False).head(14)
    s = s.iloc[::-1]
    is_social = ~s["feature"].str.contains(
        "lag_|rolling_|Temperature|month|doy|day_|week_|is_weekend|Summer|Winter|Monsoon|^Year$",
        regex=True)
    colors = [GOLD if flag else ACCENT for flag in is_social]
    lbl = [t[:26] for t in s["feature"]]
    ax.barh(lbl, s["mean_abs_shap"], color=colors, height=0.7)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    finish(ax, PRETTY[m], "Mean |SHAP| (MW)")
handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT), plt.Rectangle((0, 0), 1, 1, color=GOLD)]
fig.legend(handles, ["Calendar / weather / lag-rolling", "Socio-economic"],
           loc="lower center", ncol=2, fontsize=7.6, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Global SHAP importance, all features (top 14)", fontsize=10.5,
             weight="semibold", color=INK, y=1.01)
fig.tight_layout()
save(fig, "fig_shap_overall.png")

print("done", flush=True)
