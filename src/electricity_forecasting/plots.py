from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_eda_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
    if "Month_Name" not in frame.columns:
        frame["Month_Name"] = frame["Date"].dt.month_name()
    if "Month" not in frame.columns:
        frame["Month"] = frame["Date"].dt.month
    if "Season" not in frame.columns:
        conditions = [
            frame["Month"].isin([12, 1]),
            frame["Month"].isin([2, 3]),
            frame["Month"].isin([4, 5, 6]),
            frame["Month"].isin([7, 8, 9]),
            frame["Month"].isin([10, 11]),
        ]
        choices = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
        frame["Season"] = np.select(conditions, choices, default="unknown")
    return frame


def plot_missing_values(data: pd.DataFrame, output: str | Path) -> None:
    df = _prepare_eda_frame(data)
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values remain in the cleaned dataset.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        missing.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Missing values by column")
        ax.set_ylabel("Missing count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_missing_heatmap(data: pd.DataFrame, output: str | Path) -> None:
    df = _prepare_eda_frame(data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    if numeric_cols:
        ax.imshow(df[numeric_cols].isna().values, cmap="viridis", aspect="auto")
        ax.set_title("Missing data heatmap")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No numeric columns available for heatmap.", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_target_distribution(data: pd.DataFrame, target: str, output: str | Path) -> None:
    series = data[target].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(series, bins=40, color="steelblue", alpha=0.8)
    axes[0].set_title(f"Distribution of {target}")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Count")
    axes[1].boxplot(series.values, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
    axes[1].set_title(f"Boxplot of {target}")
    axes[1].set_ylabel(target)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_time_series_and_monthly(data: pd.DataFrame, target: str, output_dir: str | Path) -> None:
    df = _prepare_eda_frame(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series = df.set_index("Date")[target].sort_index()
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(series.index, series.values, color="navy", linewidth=1.2)
    ax.set_title(f"Daily {target} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(target)
    fig.tight_layout(); fig.savefig(output_dir / "eda_daily_demand_over_time.png", dpi=160); plt.close(fig)

    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    monthly_avg = df.groupby("Month_Name")[target].mean().reindex(month_order)
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_avg.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_title("Average demand by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average demand")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(output_dir / "eda_monthly_average_demand.png", dpi=160); plt.close(fig)

    seasonal_avg = df.groupby("Season")[target].mean().reindex(["Winter", "Spring", "Summer", "Monsoon", "Autumn"])
    fig, ax = plt.subplots(figsize=(9, 4))
    seasonal_avg.plot(kind="bar", ax=ax, color="royalblue")
    ax.set_title("Average demand by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average demand")
    fig.tight_layout(); fig.savefig(output_dir / "eda_seasonal_average_demand.png", dpi=160); plt.close(fig)


def plot_correlation_and_outliers(data: pd.DataFrame, target: str, output_dir: str | Path) -> None:
    df = _prepare_eda_frame(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation heatmap of numeric features")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(output_dir / "eda_correlation_heatmap.png", dpi=160); plt.close(fig)

        target_corr = corr[target].drop(target).sort_values(ascending=False)
        if not target_corr.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            target_corr.plot(kind="bar", ax=ax, color="darkorange")
            ax.set_title(f"Top correlations with {target}")
            ax.set_ylabel("Correlation")
            fig.tight_layout(); fig.savefig(output_dir / "eda_target_correlations.png", dpi=160); plt.close(fig)

    for column in [target, *[c for c in ["Max_Temperature", "Min_Temperature"] if c in df.columns]]:
        fig, ax = plt.subplots(figsize=(12, 5))
        df.boxplot(column=column, by="Month_Name", ax=ax, grid=False)
        ax.set_title(f"{column} boxplot by month")
        ax.set_xlabel("Month")
        ax.set_ylabel(column)
        plt.xticks(rotation=45)
        fig.tight_layout(); fig.savefig(output_dir / f"eda_{column.lower().replace(' ', '_')}_by_month.png", dpi=160); plt.close(fig)

    if "Season" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(column=target, by="Season", ax=ax, grid=False, widths=0.5)
        ax.set_title(f"Seasonal distribution of {target}")
        ax.set_xlabel("Season")
        ax.set_ylabel(target)
        fig.tight_layout(); fig.savefig(output_dir / "eda_target_by_season.png", dpi=160); plt.close(fig)

    for column in ["Max_Temperature", "Min_Temperature"]:
        if column in df.columns:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(df[column], df[target], alpha=0.5)
            ax.set_title(f"{target} vs {column}")
            ax.set_xlabel(column)
            ax.set_ylabel(target)
            fig.tight_layout(); fig.savefig(output_dir / f"eda_{target.lower().replace(' ', '_')}_vs_{column.lower()}.png", dpi=160); plt.close(fig)


def plot_yearly_social_trends(data: pd.DataFrame, target: str, output: str | Path) -> None:
    """Plot yearly trend lines for social-like annual indicators if they exist in the data."""
    df = _prepare_eda_frame(data).copy()
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
    elif "Year" not in df.columns:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No yearly social feature columns are available in this dataset.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return

    excluded = {"Date", "Year", target, "Month", "Month_Name", "Season", "day_of_year", "week_of_year", "day_of_month"}
    social_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded and col not in {"Max_Temperature", "Min_Temperature"}
    ]

    if not social_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No yearly social feature columns were found to plot.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return

    yearly = df.groupby("Year", as_index=False)[social_cols].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in social_cols:
        ax.plot(yearly["Year"], yearly[column], marker="o", linewidth=1.8, label=column)
    ax.set_title("Yearly trend of social features")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average annual value")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_eda_plots(data: pd.DataFrame, target: str, output_dir: str | Path) -> list[Path]:
    """Generate the notebook-aligned exploratory plots for the cleaned daily demand dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    plot_missing_values(data, output_dir / "eda_missing_values.png")
    outputs.append(output_dir / "eda_missing_values.png")
    plot_missing_heatmap(data, output_dir / "eda_missing_heatmap.png")
    outputs.append(output_dir / "eda_missing_heatmap.png")
    plot_target_distribution(data, target, output_dir / "eda_target_distribution.png")
    outputs.append(output_dir / "eda_target_distribution.png")
    plot_time_series_and_monthly(data, target, output_dir)
    outputs.extend([
        output_dir / "eda_daily_demand_over_time.png",
        output_dir / "eda_monthly_average_demand.png",
        output_dir / "eda_seasonal_average_demand.png",
    ])
    plot_correlation_and_outliers(data, target, output_dir)
    outputs.extend([
        output_dir / "eda_correlation_heatmap.png",
        output_dir / "eda_target_correlations.png",
        output_dir / "eda_unrestricted_demand_by_month.png",
        output_dir / "eda_max_temperature_by_month.png",
        output_dir / "eda_min_temperature_by_month.png",
        output_dir / "eda_target_by_season.png",
        output_dir / "eda_unrestricted_demand_vs_max_temperature.png",
        output_dir / "eda_unrestricted_demand_vs_min_temperature.png",
    ])
    plot_yearly_social_trends(data, target, output_dir / "eda_yearly_social_feature_trends.png")
    outputs.append(output_dir / "eda_yearly_social_feature_trends.png")
    return outputs


def plot_forecast(forecast: pd.DataFrame, model_name: str, output: str | Path) -> None:
    """Plot actual vs. predicted values over the held-out test period."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(forecast["Date"], forecast["actual"], label="Actual", color="black", linewidth=1.5)
    ax.plot(forecast["Date"], forecast["test_prediction"], label="Predicted", color="#e07b39", linewidth=1.4, linestyle="--")
    ax.set(title=f"{model_name}: test-period forecast", xlabel="Date", ylabel="Demand")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, model_name: str, output: str | Path) -> None:
    metric = "permutation_rmse_increase_mean"
    top = importance.nlargest(20, metric).sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top[metric], color="#2a6fbb")
    ax.set(title=f"{model_name}: feature sensitivity", xlabel="RMSE increase after permutation")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


def plot_builtin_feature_importance(importance: pd.DataFrame, model_name: str, output: str | Path) -> None:
    top = importance.nlargest(20, "builtin_importance").sort_values("builtin_importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["builtin_importance"], color="#e07b39")
    ax.set(title=f"{model_name}: built-in feature importance", xlabel="Importance")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


def plot_shap_summary(
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    title: str,
    output: str | Path,
    max_display: int = 20,
) -> None:
    """Beeswarm SHAP summary: per-sample feature impact magnitude and direction."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("SHAP plotting requires `pip install -e .[shap]`") from error
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_shap_summary_subset(
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    columns: list[str],
    title: str,
    output: str | Path,
    max_display: int = 20,
) -> bool:
    """`plot_shap_summary` restricted to a column subset. Returns False without
    drawing anything if none of `columns` are present in `sample`."""
    available = [column for column in columns if column in sample.columns]
    if not available:
        return False
    positions = [sample.columns.get_loc(column) for column in available]
    plot_shap_summary(sample[available], shap_values[:, positions], title, output, max_display)
    return True


def plot_social_analysis_unavailable(
    model_name: str,
    analysis_name: str,
    reason: str,
    output: str | Path,
) -> None:
    """Create an explicit chart instead of silently omitting social analysis."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.text(
        0.5,
        0.58,
        f"{model_name}: social-feature {analysis_name} unavailable",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(0.5, 0.38, reason, ha="center", va="center", wrap=True, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results.sort_values("RMSE")
    ax.bar(ordered["model"], ordered["RMSE"], color="#2a6fbb")
    ax.set(title="Model comparison", ylabel="RMSE"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
