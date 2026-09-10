"""Model-specific feature sensitivity analyses without refitting on test data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from .models import SklearnForecaster


TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
SOCIAL_FEATURES = (
    "Average Inflation (CPI) - General",
    "Average Inflation (CPI) - General__2",
    "Population in Rural Area",
    "Population in Urban Area",
    "Density of Population",
    "Life Expactancy",
    "Unemployement rate (Adjusted) (Rural)",
    "Unemployement rate (Adjusted) (Urban)",
    "Poverty Rate",
    "Per Capita Net State Domestic Product (Current Prices)",
    "Gross State Domestic Product (Current Prices)",
    "State-wise Number of Factories",
    "State-wise Total Persons Engaged",
    "State-wise Medium & Small Scale Industries - Total Number of Units",
    "Per Capita Availability of Power",
    "Installed Capacity of Power",
    "Electricity Transmission & Distribution Losses",
    "Power Requirement",
    "Net Sown Area",
)


def select_social_feature_columns(features: list[str]) -> list[str]:
    """Return Delhi Excel social features that are present in model features."""
    available = set(features)
    return [feature for feature in SOCIAL_FEATURES if feature in available]


def tree_feature_sensitivity(
    model: SklearnForecaster,
    train: pd.DataFrame,
    target: str,
    features: list[str],
    seed: int,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Rank built-in and permutation importance using training rows only."""
    usable = train.dropna(subset=features + [target])
    builtin = getattr(model.estimator, "feature_importances_", np.full(len(features), np.nan))
    permutation = permutation_importance(
        model.estimator, usable[features], usable[target], scoring="neg_root_mean_squared_error",
        n_repeats=10, random_state=seed, n_jobs=-1,
    )
    sensitivity = pd.DataFrame({
        "feature": features,
        "builtin_importance": builtin,
        "permutation_rmse_increase_mean": permutation.importances_mean,
        "permutation_rmse_increase_std": permutation.importances_std,
    })
    if selected_features is not None:
        sensitivity = sensitivity[sensitivity["feature"].isin(selected_features)]
    return sensitivity.sort_values("permutation_rmse_increase_mean", ascending=False).reset_index(drop=True)


def lstm_feature_sensitivity(model, train: pd.DataFrame, target: str, seed: int) -> pd.DataFrame:
    """Permutation sensitivity of each supervised feature column for the LSTM."""
    if model.network is None:
        raise RuntimeError("Fit LSTM before sensitivity analysis")
    try:
        import torch
    except ImportError as error:
        raise ImportError("LSTM sensitivity requires PyTorch") from error

    usable = train.dropna(subset=model.features + [target])
    feature_matrix = usable[model.features].to_numpy(dtype=np.float32)
    target_values = usable[target].to_numpy(dtype=np.float32)
    scaled_features = (feature_matrix - model.feature_mean) / model.feature_std
    lookback = model.lookback
    starts = range(lookback - 1, len(scaled_features))
    actual = target_values[lookback - 1:]

    def windows_from(matrix: np.ndarray) -> np.ndarray:
        return np.array([matrix[i - lookback + 1:i + 1] for i in starts], dtype=np.float32)

    with torch.no_grad():
        baseline = model.network(torch.tensor(windows_from(scaled_features))).numpy().ravel() * model.target_std + model.target_mean
    baseline_rmse = mean_squared_error(actual, baseline) ** 0.5

    rng, rows = np.random.default_rng(seed), []
    for index, feature in enumerate(model.features):
        shuffled = scaled_features.copy()
        shuffled[:, index] = rng.permutation(shuffled[:, index])
        with torch.no_grad():
            predicted = model.network(torch.tensor(windows_from(shuffled))).numpy().ravel() * model.target_std + model.target_mean
        rows.append({
            "feature": feature,
            "permutation_rmse_increase_mean": float(mean_squared_error(actual, predicted) ** .5 - baseline_rmse),
        })
    return pd.DataFrame(rows).sort_values("permutation_rmse_increase_mean", ascending=False).reset_index(drop=True)


def _sample_rows(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    return frame.sample(max_samples, random_state=seed).reset_index(drop=True)


def tree_shap_values(
    model: SklearnForecaster,
    train: pd.DataFrame,
    target: str,
    features: list[str],
    max_samples: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact SHAP values for a tree ensemble via `shap.TreeExplainer`, on a sample of
    training rows. Returns the sampled feature frame (for plotting) alongside a
    (samples, features) SHAP value matrix in the same column order."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("SHAP analysis requires `pip install -e .[shap]`") from error
    usable = train.dropna(subset=features + [target])
    sample = _sample_rows(usable[features], max_samples, seed)
    explainer = shap.TreeExplainer(model.estimator)
    shap_values = np.asarray(explainer.shap_values(sample))
    return sample, shap_values


def lstm_shap_values(
    model,
    train: pd.DataFrame,
    target: str,
    max_samples: int = 200,
    background_samples: int = 50,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Approximate SHAP values for the LSTM via `shap.GradientExplainer`.

    Each window's per-timestep, per-feature attribution is summed across the
    lookback dimension to give one SHAP value per feature per sample, matching the
    shape `shap.summary_plot` expects; the sampled feature frame uses each window's
    final (most recent) time step as the representative input value for coloring.
    """
    if model.network is None:
        raise RuntimeError("Fit LSTM before SHAP analysis")
    try:
        import torch
        import shap
    except ImportError as error:
        raise ImportError("LSTM SHAP analysis requires `pip install -e .[deep-learning,shap]`") from error

    usable = train.dropna(subset=model.features + [target])
    feature_matrix = usable[model.features].to_numpy(dtype=np.float32)
    scaled_features = (feature_matrix - model.feature_mean) / model.feature_std
    lookback = model.lookback
    starts = list(range(lookback - 1, len(scaled_features)))
    windows = np.array([scaled_features[i - lookback + 1:i + 1] for i in starts], dtype=np.float32)

    rng = np.random.default_rng(seed)
    if len(windows) > max_samples:
        windows = windows[rng.choice(len(windows), max_samples, replace=False)]
    background_count = min(background_samples, len(windows))
    background = torch.tensor(windows[rng.choice(len(windows), background_count, replace=False)])

    explainer = shap.GradientExplainer(model.network, background)
    raw_values = np.asarray(explainer.shap_values(torch.tensor(windows)))
    if raw_values.ndim == 4:
        raw_values = raw_values[..., 0]
    aggregated = raw_values.sum(axis=1)

    last_step_values = windows[:, -1, :] * model.feature_std + model.feature_mean
    sample = pd.DataFrame(last_step_values, columns=model.features)
    return sample, aggregated


def shap_importance_table(sample: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across the sampled rows."""
    return pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
