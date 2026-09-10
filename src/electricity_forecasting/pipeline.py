from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .data import build_dataset, build_v3_social_extension, save_dataset, selected_workbook_manifest, yearly_data_quality_report
from .features import climatological_weather, feature_columns, make_supervised, resolve_recursive_features
from .metrics import forecast_metrics
from .model_comparison import export_model_comparison
from .models import SklearnForecaster, create_model
from .plots import (
    generate_eda_plots,
    plot_builtin_feature_importance,
    plot_feature_importance,
    plot_forecast,
    plot_shap_summary,
    plot_shap_summary_subset,
    plot_social_analysis_unavailable,
)
from .sensitivity import (
    TREE_MODELS,
    lstm_feature_sensitivity,
    lstm_shap_values,
    select_social_feature_columns,
    shap_importance_table,
    tree_feature_sensitivity,
    tree_shap_values,
)
from .split import last_year_train_test_split


def prepare(config: ForecastConfig) -> pd.DataFrame:
    print(f"[pipeline] Preparing dataset from raw_dir={config.raw_dir} ...")
    data = build_dataset(config.raw_dir, config.years, config.target, config.temperature_columns, config.social_file)
    print(f"[pipeline] Built {len(data):,} rows; saving processed data to {config.processed_file} ...")
    save_dataset(data, config.processed_file)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    social_columns = data.attrs.get("social_columns", [])
    if social_columns:
        extended, forecast_log = build_v3_social_extension(data, social_columns)
        extended.to_csv(config.artifacts_dir / "social_features_extended_v3.csv", index=False)
        forecast_log.to_csv(config.artifacts_dir / "social_feature_forecast_methods_v3.csv", index=False)
        used_forecast_log = data.attrs.get("social_forecast_log")
        if used_forecast_log is not None and not used_forecast_log.empty:
            used_forecast_log.to_csv(config.artifacts_dir / "social_feature_forecast_methods_used.csv", index=False)
    selected_workbook_manifest(
        config.raw_dir, config.years, config.target, config.temperature_columns
    ).to_csv(config.artifacts_dir / "selected_workbooks.csv", index=False)
    yearly_data_quality_report(data).to_csv(config.artifacts_dir / "yearly_data_quality.csv", index=False)
    print("[pipeline] Generating exploratory analysis plots from the cleaned dataset...")
    generate_eda_plots(data, config.target, config.artifacts_dir / "eda")
    print("[pipeline] Dataset preparation complete.")
    return data


def load_processed(config: ForecastConfig) -> pd.DataFrame:
    print("[pipeline] Loading processed dataset...")
    path = config.processed_file
    if path.exists():
        data = pd.read_parquet(path)
    elif path.with_suffix(".csv").exists():
        data = pd.read_csv(path.with_suffix(".csv"))
    else:
        raise FileNotFoundError("Prepared data not found. Run the prepare command first.")
    data["Date"] = pd.to_datetime(data["Date"])
    print(f"[pipeline] Loaded {len(data):,} processed rows.")
    return data


def _recursive_predict(model, future: pd.DataFrame, history: np.ndarray, features: list[str]) -> np.ndarray:
    """Produce multi-step ML forecasts without using target values from the future period."""
    if not isinstance(model, SklearnForecaster):
        return model.predict(future)
    values = list(np.asarray(history, dtype=float))
    predictions: list[float] = []
    for _, row in future.iterrows():
        candidate = resolve_recursive_features(row, features, values)
        predicted = float(model.predict(pd.DataFrame([candidate]))[0])
        predictions.append(predicted)
        values.append(predicted)
    return np.asarray(predictions)


def train_and_evaluate(config: ForecastConfig) -> pd.DataFrame:
    """Fit once on history and evaluate only on the newest calendar year."""
    print("[pipeline] Starting train/test workflow...")
    data = load_processed(config)
    print("[pipeline] Creating supervised features for forecasting...")
    supervised = make_supervised(data, config.target, config.lags, config.rolling_windows)
    train, test = last_year_train_test_split(supervised)
    print(f"[pipeline] Train rows: {len(train):,}; Test rows: {len(test):,}")
    # Realized test-year weather is not available when a one-year-ahead forecast is
    # issued; substitute the training-period day-of-year normals so every model sees
    # only information a planner would actually hold at the forecast origin.
    test = climatological_weather(train, test, config.temperature_columns)
    print(f"[pipeline] Test-period {', '.join(config.temperature_columns)} replaced "
          f"with training day-of-year climatology")
    features = feature_columns(supervised, config.target)
    print(f"[pipeline] Using {len(features)} feature columns: {features[:10]} ...")
    if not features:
        raise ValueError("No numeric model features were created")
    if len(train.dropna(subset=features + [config.target])) < 30:
        raise ValueError("Too few complete training rows after feature engineering")
    results, all_forecasts = [], {}
    for name in config.models:
        print(f"\n[pipeline] MODEL START: {name}")
        model = create_model(name, config.seed).fit(train, config.target, features)
        print(f"[pipeline] Fitted {name}; generating test predictions...")
        test_pred = _recursive_predict(model, test, train[config.target].to_numpy(), features)
        actual = test[config.target].to_numpy()
        valid = np.isfinite(test_pred)
        if not valid.any():
            raise ValueError(f"{name} produced no valid test predictions")
        metrics = forecast_metrics(actual[valid], test_pred[valid])
        print(f"[pipeline] {name} test metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.4f}")
        results.append({"model": name, "split": "test", "observations": int(valid.sum()), **metrics})
        all_forecasts[name] = pd.DataFrame({"Date": test["Date"], "actual": actual, "test_prediction": test_pred})
        if name in TREE_MODELS:
            print(f"[pipeline] Computing feature sensitivity for {name}...")
            sensitivity = tree_feature_sensitivity(model, train, config.target, features, config.seed)
            _save_sensitivity(sensitivity, name, config)
            social_features = select_social_feature_columns(features)
            if social_features:
                print(f"[pipeline] Computing social-feature importance and sensitivity for {name} ({len(social_features)} features)...")
                social_sensitivity = tree_feature_sensitivity(
                    model, train, config.target, features, config.seed, selected_features=social_features
                )
                _save_sensitivity(social_sensitivity, name, config, suffix="_social")
                plot_builtin_feature_importance(
                    social_sensitivity,
                    f"{name}: social-feature importance",
                    config.artifacts_dir / f"feature_importance_{name}_social.png",
                )
            else:
                _save_social_analysis_unavailable(
                    name,
                    config,
                    "None of the configured social indicators are present in this model's input features.",
                )
            try:
                print(f"[pipeline] Computing SHAP values for {name}...")
                sample, shap_values = tree_shap_values(model, train, config.target, features, seed=config.seed)
                _save_shap_analysis(name, sample, shap_values, features, config)
            except ImportError as error:
                print(f"[pipeline] Skipping SHAP analysis for {name}: {error}")
        elif name == "lstm":
            print(f"[pipeline] Computing LSTM feature sensitivity for {name}...")
            sensitivity = lstm_feature_sensitivity(model, train, config.target, config.seed)
            _save_sensitivity(sensitivity, name, config)
            social_sensitivity = sensitivity[sensitivity["feature"].isin(select_social_feature_columns(features))]
            if not social_sensitivity.empty:
                _save_sensitivity(social_sensitivity, name, config, suffix="_social")
                plot_feature_importance(social_sensitivity, f"{name}: social-feature importance", config.artifacts_dir / f"feature_importance_{name}_social.png")
            else:
                _save_social_analysis_unavailable(name, config, "No configured social indicators are present in the LSTM input features.")
            try:
                print(f"[pipeline] Computing SHAP values for {name}...")
                sample, shap_values = lstm_shap_values(model, train, config.target, seed=config.seed)
                _save_shap_analysis(name, sample, shap_values, features, config)
            except ImportError as error:
                print(f"[pipeline] Skipping SHAP analysis for {name}: {error}")
        print(f"[pipeline] MODEL COMPLETE: {name}")
    results_frame = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    print("[pipeline] Saving metrics and forecast artifacts...")
    results_frame.to_csv(config.artifacts_dir / "model_metrics_test.csv", index=False)
    export_model_comparison(
        metrics_path=config.artifacts_dir / "model_metrics_test.csv",
        output_dir=config.artifacts_dir / "model_comparison_charts",
        workbook_path=config.artifacts_dir / "model_comparison.xlsx",
    )
    for name, forecast in all_forecasts.items():
        forecast.to_csv(config.artifacts_dir / f"forecast_{name}.csv", index=False)
        plot_forecast(forecast, name, config.artifacts_dir / f"forecast_{name}.png")
    results_frame.to_csv(config.artifacts_dir / "model_comparison_test.csv", index=False)
    (config.artifacts_dir / "run_metadata.json").write_text(json.dumps({
        "target": config.target, "train_start": str(train["Date"].min().date()), "train_end": str(train["Date"].max().date()),
        "test_start": str(test["Date"].min().date()), "test_end": str(test["Date"].max().date()),
        "train_observations": len(train), "test_observations": len(test), "models": list(config.models),
    }, indent=2))
    print("[pipeline] Full evaluation complete.")
    return results_frame


def _save_sensitivity(sensitivity: pd.DataFrame, name: str, config: ForecastConfig, suffix: str = "") -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"sensitivity_{name}{suffix}"
    sensitivity.to_csv(config.artifacts_dir / f"{artifact_name}.csv", index=False)
    plot_title = f"{name}: social-feature sensitivity" if suffix == "_social" else name
    plot_feature_importance(sensitivity, plot_title, config.artifacts_dir / f"{artifact_name}.png")
    top = sensitivity.head(5)
    summary = "; ".join(f"{row.feature} ({row.permutation_rmse_increase_mean:.4f} RMSE increase)" for row in top.itertuples())
    description = "Delhi social features" if suffix == "_social" else "features"
    (config.artifacts_dir / f"{artifact_name}_interpretation.txt").write_text(
        f"Most influential {description} for {name}, ranked by training-period permutation sensitivity: {summary}.\n"
    )


def _save_social_analysis_unavailable(name: str, config: ForecastConfig, reason: str) -> None:
    """Persist a transparent social-analysis status for models without social inputs."""
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"sensitivity_{name}_social"
    pd.DataFrame([{
        "model": name,
        "analysis": "social feature importance and permutation sensitivity",
        "status": "unavailable",
        "reason": reason,
    }]).to_csv(config.artifacts_dir / f"{artifact_name}.csv", index=False)
    plot_social_analysis_unavailable(name, "importance", reason, config.artifacts_dir / f"feature_importance_{name}_social.png")
    plot_social_analysis_unavailable(name, "sensitivity", reason, config.artifacts_dir / f"{artifact_name}.png")
    (config.artifacts_dir / f"{artifact_name}_interpretation.txt").write_text(f"Social-feature analysis unavailable for {name}. {reason}\n")


def _save_shap_analysis(
    name: str,
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    features: list[str],
    config: ForecastConfig,
) -> None:
    """Save SHAP importance tables/plots for all features and, if present, the social subset."""
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    importance = shap_importance_table(sample, shap_values)
    importance.to_csv(config.artifacts_dir / f"shap_values_{name}.csv", index=False)
    plot_shap_summary(sample, shap_values, f"{name}: SHAP summary (top features)", config.artifacts_dir / f"shap_summary_{name}.png")

    social_features = select_social_feature_columns(features)
    social_output = config.artifacts_dir / f"shap_summary_{name}_social.png"
    if social_features and plot_shap_summary_subset(sample, shap_values, social_features, f"{name}: SHAP summary (social features)", social_output):
        importance[importance["feature"].isin(social_features)].reset_index(drop=True).to_csv(
            config.artifacts_dir / f"shap_values_{name}_social.csv", index=False
        )
    else:
        reason = "None of the configured social indicators are present in this model's SHAP sample."
        plot_social_analysis_unavailable(name, "SHAP summary", reason, social_output)
