from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ("MAE", "MSE", "RMSE", "MAPE", "sMAPE", "WAPE", "MBE", "R2", "MASE")
HIGHER_IS_BETTER = {"R2"}

# MAE/MSE/RMSE/MAPE/sMAPE/WAPE/MASE are all just different normalizations of the same
# underlying question — "how big is the error" — so treating absolute/percentage/
# relative error as three separate top-level groups still gives that one question 60%
# of the total weight (3 groups x equal share) against 20% each for bias (MBE) and fit
# (R2). This taxonomy is two-level instead: "error magnitude" is ONE top-level concept
# (on equal footing with bias and fit, so all three get 1/3), subdivided into its three
# normalizations (absolute/percentage/relative), each split evenly among its member
# metrics — so no single normalization of "error size" dominates the other two either.
METRIC_TAXONOMY = {
    "MAE": ("error_magnitude", "absolute_error"), "MSE": ("error_magnitude", "absolute_error"), "RMSE": ("error_magnitude", "absolute_error"),
    "MAPE": ("error_magnitude", "percentage_error"), "sMAPE": ("error_magnitude", "percentage_error"), "WAPE": ("error_magnitude", "percentage_error"),
    "MASE": ("error_magnitude", "relative_error"),
    "MBE": ("bias", "bias"),
    "R2": ("fit", "fit"),
}


def default_metric_weights(metric_columns: list[str]) -> dict[str, float]:
    """Equal weight per top-level concept (error magnitude / bias / fit); within
    error magnitude, equal weight per normalization (absolute/percentage/relative);
    within a normalization, equal weight per metric present in `metric_columns`.
    A metric with no assigned taxonomy entry is its own top-level group of one."""
    tree: dict[str, dict[str, list[str]]] = {}
    for metric in metric_columns:
        top, sub = METRIC_TAXONOMY.get(metric, (metric, metric))
        tree.setdefault(top, {}).setdefault(sub, []).append(metric)
    top_weight = 1.0 / len(tree)
    weights: dict[str, float] = {}
    for subgroups in tree.values():
        sub_weight = top_weight / len(subgroups)
        for members in subgroups.values():
            member_weight = sub_weight / len(members)
            weights.update({metric: member_weight for metric in members})
    return weights


def _normalise_metric(values: pd.Series, metric: str) -> pd.Series:
    """Scale a metric to [0, 1], where 1 always means best performance."""
    numeric = pd.to_numeric(values, errors="coerce")
    if metric == "MBE":
        numeric = numeric.abs()
    minimum, maximum = numeric.min(), numeric.max()
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.nan, index=values.index, dtype=float)
    if maximum == minimum:
        return pd.Series(1.0, index=values.index, dtype=float)
    if metric in HIGHER_IS_BETTER:
        return (numeric - minimum) / (maximum - minimum)
    return (maximum - numeric) / (maximum - minimum)


def build_period_comparison(
    metrics: pd.DataFrame,
    split: str,
    metric_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and normalized/composite comparisons for one split only."""
    period = metrics.loc[metrics["split"].eq(split)].copy()
    if period.empty:
        raise ValueError(f"No rows found for split={split!r}")

    metric_columns = [column for column in DEFAULT_METRICS if column in period.columns]
    if not metric_columns:
        raise ValueError("No recognized error metrics found in the metrics table")
    period = period[["model", "split", "observations", *metric_columns]].drop_duplicates("model")

    weights = metric_weights or default_metric_weights(metric_columns)
    weights = {metric: float(weights.get(metric, 0.0)) for metric in metric_columns}
    if sum(weights.values()) <= 0:
        raise ValueError("Metric weights must have a positive total")
    weights = {metric: weight / sum(weights.values()) for metric, weight in weights.items()}

    normalized = period[["model", "split", "observations"]].copy()
    for metric in metric_columns:
        normalized[f"{metric}_normalized"] = _normalise_metric(period[metric], metric)
    normalized["composite_score"] = sum(
        normalized[f"{metric}_normalized"] * weight for metric, weight in weights.items()
    )
    normalized["rank"] = normalized["composite_score"].rank(method="min", ascending=False).astype(int)
    normalized = normalized.sort_values(["rank", "model"]).reset_index(drop=True)
    return period.sort_values("model").reset_index(drop=True), normalized


def _plot_metric(period: pd.DataFrame, metric: str, split: str, output_dir: Path) -> None:
    ordered = period.sort_values(metric, ascending=metric in HIGHER_IS_BETTER)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(ordered["model"], ordered[metric], color="#176b87")
    axis.set_title(f"{split.title()} model comparison: {metric}")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / f"{split}_{metric.lower()}.png", dpi=160)
    plt.close(figure)


def _plot_composite(normalized: pd.DataFrame, split: str, output_dir: Path) -> None:
    ordered = normalized.sort_values(["composite_score", "model"], ascending=[False, True])
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(ordered["model"], ordered["composite_score"], color="#d35f3f")
    axis.set_title(f"{split.title()} composite performance ranking")
    axis.set_xlabel("Model")
    axis.set_ylabel("Composite score (0 = worst, 1 = best)")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / f"{split}_composite_score.png", dpi=160)
    plt.close(figure)


def export_model_comparison(
    metrics_path: str | Path,
    output_dir: str | Path,
    workbook_path: str | Path,
    metric_weights: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Export separate train/test tables and charts from pipeline metrics."""
    metrics = pd.read_csv(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = Path(workbook_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    splits = [split for split in ("train", "test") if split in set(metrics["split"])]
    if not splits:
        raise ValueError("The metrics artifact contains neither train nor test rows")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        methodology = pd.DataFrame({
            "item": ["Metrics", "Normalization", "Direction", "Weights", "Composite score"],
            "detail": [
                "All metric columns present in the artifact are included.",
                "For each split and metric, min-max scaling is applied independently: (max - value) / (max - min) for errors; (value - min) / (max - min) for R2.",
                "MAE, MSE, RMSE, MAPE, sMAPE, WAPE, and MASE are lower-is-better; MBE is scored by absolute distance from zero; R2 is higher-is-better.",
                "Unless metric_weights is supplied: error magnitude, bias (MBE), and fit (R2) each get equal weight (1/3) as top-level concepts; within error magnitude, absolute (MAE/MSE/RMSE), percentage (MAPE/sMAPE/WAPE), and relative-to-naive (MASE) normalizations each get an equal share of that third, split evenly across their member metrics present in the artifact. This avoids the many correlated error-size metrics diluting bias and fit, at any level. Weights are renormalized to sum to 1.",
                "Weighted sum of normalized metric scores; 1 is best within a split. Rankings sort descending by composite score.",
            ],
        })
        methodology.to_excel(writer, sheet_name="Methodology", index=False)

        for split in splits:
            raw, normalized = build_period_comparison(metrics, split, metric_weights)
            raw.to_excel(writer, sheet_name=f"{split.title()} Metrics", index=False)
            normalized.to_excel(writer, sheet_name=f"{split.title()} Scores", index=False)
            outputs[f"{split}_metrics"] = raw
            outputs[f"{split}_scores"] = normalized
            for metric in [column for column in DEFAULT_METRICS if column in raw.columns]:
                _plot_metric(raw, metric, split, output_dir)
            _plot_composite(normalized, split, output_dir)
    return outputs
