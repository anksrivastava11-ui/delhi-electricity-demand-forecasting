"""Shared interfaces and the factory for one-file-per-method forecasters."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Forecaster(ABC):
    @abstractmethod
    def fit(self, train: pd.DataFrame, target: str, features: list[str]) -> "Forecaster": ...

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> np.ndarray: ...


class SklearnForecaster(Forecaster):
    """Shared training and inference adapter for sklearn-compatible estimators."""
    def __init__(self, estimator) -> None:
        self.estimator, self.features = estimator, []

    def fit(self, train: pd.DataFrame, target: str, features: list[str]) -> "SklearnForecaster":
        self.features = features
        usable = train.dropna(subset=features + [target])
        self.estimator.fit(usable[features], usable[target])
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(future[self.features])


def create_model(name: str, seed: int = 42) -> Forecaster:
    """Instantiate a named method; implementations are deliberately isolated."""
    name = name.lower()
    if name == "seasonal_naive":
        from .forecasters.seasonal_naive import SeasonalNaiveForecaster
        return SeasonalNaiveForecaster()
    if name == "holt_winters":
        from .forecasters.holt_winters import HoltWintersForecaster
        return HoltWintersForecaster()
    if name == "arima":
        from .forecasters.arima import ARIMAForecaster
        return ARIMAForecaster()
    if name == "sarimax":
        from .forecasters.sarimax import SARIMAXForecaster
        return SARIMAXForecaster()
    if name == "prophet":
        from .forecasters.prophet_model import ProphetForecaster
        return ProphetForecaster()
    if name == "random_forest":
        from .forecasters.random_forest import RandomForestForecaster
        return RandomForestForecaster(seed)
    if name == "xgboost":
        from .forecasters.xgboost_model import XGBoostForecaster
        return XGBoostForecaster(seed)
    if name == "lightgbm":
        from .forecasters.lightgbm_model import LightGBMForecaster
        return LightGBMForecaster(seed)
    if name == "lstm":
        from .forecasters.lstm import LSTMForecaster
        return LSTMForecaster(seed=seed)
    raise ValueError(f"Unsupported model: {name}")
