from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForecastConfig:
    raw_dir: Path = Path("data/raw")
    processed_file: Path = Path("data/processed/daily_demand.parquet")
    artifacts_dir: Path = Path("artifacts")
    target: str = "Unrestricted demand"
    date_column: str = "Date"
    temperature_columns: tuple[str, ...] = ("Max_Temperature", "Min_Temperature")
    social_file: Path | None = Path("../Delhi.xlsx")
    years: tuple[int, ...] = tuple(range(2014, 2024))
    lags: tuple[int, ...] = (1, 2, 3, 7, 14, 28, 365)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    models: tuple[str, ...] = ("seasonal_naive", "holt_winters", "random_forest")
    seed: int = 42


def load_config(path: str | Path) -> ForecastConfig:
    """Load a YAML config, ignoring no fields and failing early on unknown keys."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    paths, data = raw.get("paths", {}), raw.get("data", {})
    features = raw.get("features", {})
    known = {"paths", "data", "features", "models", "seed"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown configuration section(s): {sorted(unknown)}")
    return ForecastConfig(
        raw_dir=Path(paths.get("raw_dir", "data/raw")),
        processed_file=Path(paths.get("processed_file", "data/processed/daily_demand.parquet")),
        artifacts_dir=Path(paths.get("artifacts_dir", "artifacts")),
        target=data.get("target", "Unrestricted demand"),
        date_column=data.get("date_column", "Date"),
        temperature_columns=tuple(data.get("temperature_columns", ["Max_Temperature", "Min_Temperature"])),
        social_file=Path(data["social_file"]) if data.get("social_file") else None,
        years=tuple(data.get("years", range(2014, 2024))),
        lags=tuple(features.get("lags", [1, 7, 14, 28, 365])),
        rolling_windows=tuple(features.get("rolling_windows", [7, 14, 28])),
        models=tuple(raw.get("models", ["seasonal_naive", "holt_winters", "random_forest"])),
        seed=int(raw.get("seed", 42)),
    )
