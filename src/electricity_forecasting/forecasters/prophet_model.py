"""Meta Prophet forecaster."""
import numpy as np
import pandas as pd
from ..models import Forecaster


class ProphetForecaster(Forecaster):
    def __init__(self) -> None: self.model = None
    def fit(self, train: pd.DataFrame, target: str, features: list[str]):
        try: from prophet import Prophet
        except ImportError as error: raise ImportError("Prophet requires `pip install -e .[prophet]`") from error
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.model.fit(train[["Date", target]].rename(columns={"Date": "ds", target: "y"})); return self
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model is not fitted")
        return self.model.predict(future[["Date"]].rename(columns={"Date": "ds"}))["yhat"].to_numpy()
