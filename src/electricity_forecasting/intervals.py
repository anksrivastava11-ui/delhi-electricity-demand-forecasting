from __future__ import annotations

import numpy as np


class ConformalInterval:
    """Distribution-free symmetric interval calibrated on a separate historical period."""
    def __init__(self, coverage: float = 0.95) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must lie strictly between zero and one")
        self.coverage = coverage
        self.radius: float | None = None

    def fit(self, actual: np.ndarray, predicted: np.ndarray) -> "ConformalInterval":
        errors = np.abs(np.asarray(actual, float) - np.asarray(predicted, float))
        n = len(errors)
        if n < 10:
            raise ValueError("At least 10 calibration residuals are required")
        quantile = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.radius = float(np.quantile(errors, quantile, method="higher"))
        return self

    def predict(self, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.radius is None:
            raise RuntimeError("Fit intervals before predicting")
        forecast = np.asarray(forecast, float)
        return forecast - self.radius, forecast + self.radius
