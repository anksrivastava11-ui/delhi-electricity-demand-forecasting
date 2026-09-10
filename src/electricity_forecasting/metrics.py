from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, seasonal_period: int = 7) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = predicted - actual
    denominator = np.maximum(np.abs(actual), 1e-8)
    mae = mean_absolute_error(actual, predicted)
    scale = np.mean(np.abs(np.diff(actual, n=1))) if len(actual) > 1 else np.nan
    return {
        "MAE": float(mae),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(np.mean(np.abs(error) / denominator) * 100),
        "sMAPE": float(np.mean(2 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)) * 100),
        "WAPE": float(np.sum(np.abs(error)) / np.maximum(np.sum(np.abs(actual)), 1e-8) * 100),
        "MBE": float(np.mean(error)),
        "R2": float(r2_score(actual, predicted)),
        "MASE": float(mae / scale) if scale and not np.isnan(scale) else np.nan,
    }


def interval_metrics(actual: pd.Series | np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    actual, lower, upper = map(lambda item: np.asarray(item, dtype=float), (actual, lower, upper))
    return {"interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))), "mean_interval_width": float(np.mean(upper - lower))}
