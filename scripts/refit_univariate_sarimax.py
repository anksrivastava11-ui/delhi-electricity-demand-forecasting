"""Refit the univariate SARIMAX ablation variant and export its test-year forecast series.

Mirrors forecasters/sarimax.py exactly except that no exogenous Fourier terms are
supplied, so the resulting series can be plotted alongside the shipped exogenous
variant in the paper's ablation figure.
"""
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE = Path(__file__).resolve().parents[1]
TARGET = "Unrestricted demand"
TRAIN_END = "2022-07-31"
TEST_START = "2022-08-01"

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


df = load_processed_daily().sort_values("Date").reset_index(drop=True)

train = df[df["Date"] <= TRAIN_END]
test = df[df["Date"] >= TEST_START]
print(f"train={len(train)} test={len(test)}", flush=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = SARIMAX(
        train[TARGET].astype(float),
        order=(3, 1, 3),
        seasonal_order=(2, 0, 1, 7),
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=300)

pred = np.asarray(res.forecast(len(test)))
out = pd.DataFrame(
    {"Date": test["Date"].to_numpy(), "actual": test[TARGET].to_numpy(), "test_prediction": pred}
)
path = BASE / "artifacts" / "sarimax_univariate_forecast.csv"
path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)

err = pred - out["actual"].to_numpy()
rmse = float(np.sqrt(np.mean(err**2)))
mae = float(np.mean(np.abs(err)))
mape = float(np.mean(np.abs(err) / out["actual"].to_numpy()) * 100)
ss_res = float(np.sum(err**2))
ss_tot = float(np.sum((out["actual"].to_numpy() - out["actual"].mean()) ** 2))
print(f"RMSE={rmse:.2f} MAE={mae:.2f} MAPE={mape:.2f} R2={1 - ss_res / ss_tot:.3f}", flush=True)
print("wrote", path, flush=True)
