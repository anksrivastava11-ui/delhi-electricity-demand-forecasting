"""ARIMA forecaster."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from ..models import Forecaster


class ARIMAForecaster(Forecaster):
    def __init__(self, order: tuple[int, int, int] = (3, 1, 3), trend: str = "t") -> None:
        # trend="t" adds a deterministic linear drift so multi-step forecasts keep
        # extrapolating the series' long-run direction instead of flattening to the
        # last training level once the AR/MA terms decay.
        self.order, self.trend, self.model = order, trend, None
    def fit(self, train: pd.DataFrame, target: str, features: list[str]):
        self.model = ARIMA(train[target].astype(float), order=self.order, trend=self.trend).fit(); return self
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model is not fitted")
        return np.asarray(self.model.forecast(len(future)))
