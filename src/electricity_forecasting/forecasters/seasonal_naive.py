"""Seasonal-naive forecasting baseline."""
import numpy as np
import pandas as pd
from ..models import Forecaster


class SeasonalNaiveForecaster(Forecaster):
    def __init__(self, period: int = 365) -> None:
        self.period, self.history = period, None
    def fit(self, train: pd.DataFrame, target: str, features: list[str]):
        self.history = train[target].to_numpy(float); return self
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.history is None: raise RuntimeError("Model is not fitted")
        return np.resize(self.history[-self.period:], len(future))
