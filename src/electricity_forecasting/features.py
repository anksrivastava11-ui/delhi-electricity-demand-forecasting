from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    date = pd.to_datetime(data["Date"])
    data["day_of_week"] = date.dt.dayofweek
    data["day_of_month"] = date.dt.day
    data["month"] = date.dt.month
    data["day_of_year"] = date.dt.dayofyear
    data["week_of_year"] = date.dt.isocalendar().week.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["month_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * data["month"] / 12)
    data["month_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * data["month"] / 12)
    data["doy_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * data["day_of_year"] / 365.25)
    data["doy_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * data["day_of_year"] / 365.25)
    return data


def make_supervised(frame: pd.DataFrame, target: str, lags: tuple[int, ...], rolling_windows: tuple[int, ...]) -> pd.DataFrame:
    """Create features with shifted targets only; each row is safe for a one-step forecast."""
    data = add_calendar_features(frame).sort_values("Date").reset_index(drop=True)
    for lag in lags:
        data[f"lag_{lag}"] = data[target].shift(lag)
    for window in rolling_windows:
        history = data[target].shift(1)
        data[f"rolling_mean_{window}"] = history.rolling(window).mean()
        data[f"rolling_std_{window}"] = history.rolling(window).std()
    return data


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {"Date", target}
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def climatological_weather(
    train: pd.DataFrame,
    future: pd.DataFrame,
    temperature_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Replace future weather with a training-period day-of-year climatology.

    Realized temperature for a future year is not knowable at the moment a
    one-year-ahead forecast is issued, so feeding observed test-period temperature
    to a model leaks information that will not exist in deployment. We substitute
    the mean of each calendar day across the training years -- the standard
    "normal" a planner does have in advance. Day 366 falls back to day 365, and any
    day-of-year absent from training falls back to the training mean.

    Non-positive readings are excluded from the averages: a few rows carry a
    sentinel 0 rather than a real observation, which would drag the normal down.
    """
    future = future.copy()
    train = add_calendar_features(train) if "day_of_year" not in train else train
    if "day_of_year" not in future:
        future = add_calendar_features(future)

    for column in temperature_columns:
        if column not in future.columns or column not in train.columns:
            continue
        observed = train.loc[train[column] > 0, ["day_of_year", column]]
        normals = observed.groupby("day_of_year")[column].mean()
        overall = float(observed[column].mean())
        day = future["day_of_year"].replace({366: 365})
        future[column] = day.map(normals).astype(float).fillna(overall)
    return future


def resolve_recursive_features(row: pd.Series, features: list[str], values: list[float]) -> pd.Series:
    """Fill target-derived lag/rolling columns from a running history of true/predicted values.

    Calendar and social columns already present on `row` are left untouched since they
    are known ahead of time and do not depend on the target.
    """
    candidate = row.copy()
    for feature in features:
        if feature.startswith("lag_"):
            lag = int(feature.removeprefix("lag_"))
            candidate[feature] = values[-lag] if len(values) >= lag else np.nan
        elif feature.startswith("rolling_mean_"):
            window = int(feature.removeprefix("rolling_mean_"))
            candidate[feature] = np.mean(values[-window:]) if values else np.nan
        elif feature.startswith("rolling_std_"):
            window = int(feature.removeprefix("rolling_std_"))
            candidate[feature] = np.std(values[-window:], ddof=1) if len(values) > 1 else 0.0
    return candidate
