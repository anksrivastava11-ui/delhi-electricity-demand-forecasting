"""Holt-Winters exponential-smoothing forecaster."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..models import Forecaster


class HoltWintersForecaster(Forecaster):
    def __init__(self, period: int = 365, damped_trend: bool = True) -> None:
        self.period = period
        self.damped_trend = damped_trend
        self.model = None
    def fit(self, train: pd.DataFrame, target: str, features: list[str]):
        self.model = ExponentialSmoothing(
            train[target].astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=self.period,
            damped_trend=self.damped_trend,
        ).fit(optimized=True)
        return self
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model is not fitted")
        return np.asarray(self.model.forecast(len(future)))
