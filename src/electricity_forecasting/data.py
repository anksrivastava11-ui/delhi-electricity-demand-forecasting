from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.holtwinters import Holt

MONTHS = {name: number for number, name in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
)}


def read_year_workbook(path: str | Path, year: int, day_column: str = "Date") -> pd.DataFrame:
    """Read month-named sheets and construct a genuine daily date column."""
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []
    for name, frame in sheets.items():
        key = str(name).strip().upper()[:3]
        if key not in MONTHS or day_column not in frame:
            continue
        result = frame.copy()
        days = pd.to_numeric(result[day_column], errors="coerce")
        result["Date"] = pd.to_datetime({"year": year, "month": MONTHS[key], "day": days}, errors="coerce")
        frames.append(result)
    if not frames:
        raise ValueError(f"No month sheets containing {day_column!r} found in {path}")
    return pd.concat(frames, ignore_index=True)


def _workbook_score(path: Path, day_column: str, required_columns: tuple[str, ...]) -> tuple[int, int, int]:
    """Score a workbook by usable month sheets, then by available metadata.

    Some years have more than one workbook.  Selecting alphabetically silently
    chose the abbreviated 2018 workbook, which discarded the seasonal fields.
    """
    sheets = pd.read_excel(path, sheet_name=None, nrows=0)
    usable = [
        frame for name, frame in sheets.items()
        if str(name).strip().upper()[:3] in MONTHS and day_column in frame.columns
    ]
    required_matches = sum(sum(column in frame.columns for column in required_columns) for frame in usable)
    total_columns = sum(len(frame.columns) for frame in usable)
    return len(usable), required_matches, total_columns


def _find_year_file(
    raw_dir: Path,
    year: int,
    day_column: str = "Date",
    required_columns: tuple[str, ...] = (),
) -> Path:
    candidates = sorted(p for p in raw_dir.glob(f"*{year}*.xls*") if not p.name.startswith("~$"))
    if not candidates:
        raise FileNotFoundError(f"No workbook for {year} in {raw_dir}")
    return max(candidates, key=lambda path: _workbook_score(path, day_column, required_columns))


def selected_workbook_manifest(
    raw_dir: str | Path,
    years: tuple[int, ...],
    target: str,
    temperature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Document exactly which annual file was selected and why."""
    root = Path(raw_dir)
    rows = []
    required = (target, *temperature_columns)
    for year in years:
        path = _find_year_file(root, year, required_columns=required)
        usable_sheets, required_matches, total_columns = _workbook_score(path, "Date", required)
        rows.append({
            "year": year,
            "workbook": path.name,
            "usable_month_sheets": usable_sheets,
            "required_column_matches": required_matches,
            "total_columns_across_month_sheets": total_columns,
        })
    return pd.DataFrame(rows)


SEASON_MONTHS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Pre Monsoon", 11: "Pre Monsoon",
}
SEASON_COLUMNS = ("Summer", "Winter", "Monsoon", "Pre Monsoon")


def fill_season_flags_by_calendar_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Summer/Winter/Monsoon/Pre Monsoon flags from each row's calendar month.

    Every non-missing row in this dataset already follows this exact one-hot
    month-to-season mapping (Winter=Dec-Feb, Summer=Mar-Jun, Monsoon=Jul-Sep,
    Pre Monsoon=Oct-Nov; verified with zero exceptions across ~3,300 rows) — some
    source month sheets simply have no row for their 31st day, so a handful of dates
    are missing these flags after reindexing. Filled the same way the rest of the
    column already is, rather than left NaN and silently dropped from training.
    """
    data = frame.copy()
    available = [column for column in SEASON_COLUMNS if column in data.columns]
    if not available:
        return data
    missing = data[available].isna().any(axis=1)
    if missing.any():
        season = data.loc[missing, "Date"].dt.month.map(SEASON_MONTHS)
        for column in available:
            data.loc[missing, column] = (season == column).astype(float)
    return data


def clean_daily_data(frame: pd.DataFrame, target: str, temperature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Standardise numeric columns, remove duplicates, and preserve missing days explicitly."""
    if "Date" not in frame or target not in frame:
        raise ValueError(f"Expected Date and target {target!r}; got {frame.columns.tolist()}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in [target, *temperature_columns]:
        if column in data:
            data[column] = pd.to_numeric(data[column].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    data = data.dropna(subset=["Date", target]).sort_values("Date")
    data = data.groupby("Date", as_index=False).mean(numeric_only=True)
    index = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    data = data.set_index("Date").reindex(index).rename_axis("Date").reset_index()
    data[target] = data[target].interpolate(limit_direction="both")
    for column in temperature_columns:
        if column in data:
            data[column] = data[column].interpolate(limit_direction="both")
    data = fill_season_flags_by_calendar_month(data)
    if data[target].isna().any():
        raise ValueError("Target remains missing after cleaning")
    return data


def load_notebook_delhi_social_features(path: str | Path) -> pd.DataFrame:
    """Reproduce the notebook's Delhi.xlsx transpose and fiscal-to-calendar year alignment.

    Returns only the years genuinely present in the source workbook — callers needing
    years beyond that range should extrapolate explicitly (see `extend_social_features_by_trend`)
    rather than have this silently repeat the last known year forward.
    """
    social = pd.read_excel(path).T.reset_index()
    social.columns = social.iloc[0]
    social = social.drop(0).rename(columns={"Unnamed: 0": "Year"}).reset_index(drop=True)
    social = social.drop(0).copy()
    # Delhi.xlsx contains "Average Inflation (CPI) - General" twice.  Preserve
    # both notebook-derived values while giving the second a unique dataframe
    # label required by sklearn feature matrices.
    seen: dict[str, int] = {}
    columns = []
    for column in social.columns:
        label = str(column)
        seen[label] = seen.get(label, 0) + 1
        columns.append(label if seen[label] == 1 else f"{label}__{seen[label]}")
    social.columns = columns
    social["Year"] = social["Year"].apply(lambda value: int(str(value).split("-")[0]) + 1)
    for column in social.columns:
        if column != "Year":
            social[column] = pd.to_numeric(social[column], errors="coerce")
    return social


def forecast_social_variable_damped(series: pd.Series, steps: int) -> tuple[np.ndarray, str]:
    """Forecast a short annual series forward with Holt's damped-trend exponential
    smoothing. The damping parameter is fit from the data itself (not a fixed cap), and
    a damped trend mathematically tapers off over a multi-step horizon rather than
    extrapolating a straight line/exponential indefinitely — the right tool for
    projecting a handful of annual points a few years out without manufacturing an
    implausible swing (e.g. a percentage-rate variable projected past 100%)."""
    values = series.to_numpy(dtype=float)
    if len(values) < 4 or np.ptp(values) == 0:
        return np.full(steps, values[-1]), "constant"
    try:
        fitted = Holt(values, damped_trend=True, initialization_method="estimated").fit(optimized=True)
        forecast = np.asarray(fitted.forecast(steps))
        if not np.all(np.isfinite(forecast)):
            raise ValueError("non-finite damped-trend forecast")
        return forecast, "damped_trend"
    except Exception:
        return np.full(steps, values[-1]), "constant_fallback"


def extend_social_features_by_trend(social: pd.DataFrame, missing_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill social-feature years absent from the source workbook using each variable's own
    damped-trend forecast (`forecast_social_variable_damped`), rather than repeating the
    last known year forward. Each column is forecast once, as a single multi-step-ahead
    horizon from its real annual values, matching how the damping actually tapers a
    trend over consecutive years rather than treating each missing year independently."""
    columns = [column for column in social.columns if column != "Year"]
    indexed = social.set_index("Year")
    last_year = int(indexed.index.max())
    horizon = max(missing_years) - last_year
    forecasts: dict[str, np.ndarray] = {}
    methods: dict[str, str] = {}
    for column in columns:
        series = indexed[column].dropna()
        forecasts[column], methods[column] = forecast_social_variable_damped(series, horizon)
    rows, log = [], []
    for year in missing_years:
        row = {"Year": year}
        offset = year - last_year - 1
        for column in columns:
            row[column] = float(forecasts[column][offset])
            log.append({"feature": column, "year": year, "method": methods[column]})
        rows.append(row)
    extended = pd.concat([social, pd.DataFrame(rows)], ignore_index=True).sort_values("Year").reset_index(drop=True)
    return extended, pd.DataFrame(log, columns=["feature", "year", "method"])


def merge_notebook_delhi_social_features(frame: pd.DataFrame, social_file: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-merge notebook-aligned annual Delhi features onto every daily row.

    Years the daily data needs but the source workbook doesn't cover are filled by
    per-variable trend extrapolation (`extend_social_features_by_trend`) instead of
    repeating the last known year — the source workbook's real coverage stops well
    before the dataset's later years, so without this the most recent years (including
    the whole test/forecast horizon) would otherwise see a frozen repeat of stale data.
    """
    data = frame.copy()
    data["Year"] = pd.to_datetime(data["Date"]).dt.year
    social = load_notebook_delhi_social_features(social_file)
    needed_years = sorted(data["Year"].unique())
    missing_years = [year for year in needed_years if year not in set(social["Year"])]
    forecast_log = pd.DataFrame(columns=["feature", "year", "method"])
    if missing_years:
        social, forecast_log = extend_social_features_by_trend(social, missing_years)
    merged = data.merge(social, on="Year", how="left", validate="many_to_one")
    social_columns = [column for column in social.columns if column != "Year"]
    if merged[social_columns].isna().any().any():
        remaining = sorted(merged.loc[merged[social_columns].isna().any(axis=1), "Year"].unique())
        raise ValueError(f"Delhi social features are missing for year(s): {remaining}")
    return merged, forecast_log


def forecast_social_variable_v3(series_by_year: pd.Series, future_years: list[int]) -> tuple[dict[int, float], str]:
    """Match v3: select linear or exponential annual trend by in-sample RMSE."""
    years = np.asarray(series_by_year.index, dtype=float)
    values = np.asarray(series_by_year.values, dtype=float)
    slope_lin, intercept_lin, *_ = linregress(years, values)
    rmse_lin = float(np.sqrt(np.mean((values - (slope_lin * years + intercept_lin)) ** 2)))
    slope_exp = intercept_exp = None
    rmse_exp = np.inf
    if np.all(values > 0):
        slope_exp, intercept_exp, *_ = linregress(years, np.log(values))
        rmse_exp = float(np.sqrt(np.mean((values - np.exp(slope_exp * years + intercept_exp)) ** 2)))
    use_exp = rmse_exp < rmse_lin and slope_exp is not None
    forecasts = {
        year: float(np.exp(slope_exp * year + intercept_exp) if use_exp else slope_lin * year + intercept_lin)
        for year in future_years
    }
    return forecasts, "exponential" if use_exp else "linear"


def build_v3_social_extension(data: pd.DataFrame, social_columns: list[str], years_ahead: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce v3's annual social aggregation and per-variable five-year forecast."""
    yearly = data.groupby("Year", as_index=False)[social_columns].mean().sort_values("Year")
    future_years = list(range(int(yearly["Year"].max()) + 1, int(yearly["Year"].max()) + years_ahead + 1))
    rows, log = [], []
    for year in future_years:
        row = {"Year": year}
        for column in social_columns:
            values = yearly.set_index("Year")[column].dropna()
            if len(values) < 2:
                row[column], method = float(values.iloc[-1]), "insufficient_data"
            else:
                forecast, method = forecast_social_variable_v3(values, [year])
                row[column] = forecast[year]
            if year == future_years[0]:
                log.append({"feature": column, "method": method})
        rows.append(row)
    return pd.concat([yearly, pd.DataFrame(rows)], ignore_index=True), pd.DataFrame(log)


def build_dataset(raw_dir: str | Path, years: tuple[int, ...], target: str, temperature_columns: tuple[str, ...], social_file: str | Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = [
        read_year_workbook(
            _find_year_file(raw_dir, year, required_columns=(target, *temperature_columns)), year
        )
        for year in years
    ]
    data = clean_daily_data(pd.concat(frames, ignore_index=True), target, temperature_columns)
    if not social_file:
        return data
    merged, forecast_log = merge_notebook_delhi_social_features(data, social_file)
    merged.attrs["social_columns"] = [column for column in load_notebook_delhi_social_features(social_file).columns if column != "Year"]
    merged.attrs["social_forecast_log"] = forecast_log
    return merged


def yearly_data_quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an auditable, per-year completeness report for every data column."""
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["year"] = data["Date"].dt.year
    value_columns = [column for column in data.columns if column not in {"Date", "year"}]
    rows = []
    for year, group in data.groupby("year", sort=True):
        for column in value_columns:
            rows.append({
                "year": int(year),
                "column": column,
                "rows": len(group),
                "missing_values": int(group[column].isna().sum()),
                "missing_percent": round(float(group[column].isna().mean() * 100), 3),
            })
    return pd.DataFrame(rows)


def save_dataset(frame: pd.DataFrame, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        frame.to_csv(path.with_suffix(".csv"), index=False)
