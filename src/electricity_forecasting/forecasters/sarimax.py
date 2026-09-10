"""Weekly + annual seasonal SARIMAX forecaster."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..models import Forecaster


class SARIMAXForecaster(Forecaster):

    def __init__(
        self,
        order: tuple[int, int, int] = (3, 1, 3),
        seasonal_period: int = 7,
        annual_harmonics: int = 3,
    ) -> None:
        # A seasonal_order period of 365 is computationally impractical for statsmodels'
        # state-space SARIMAX on a multi-year daily series (state size scales with the
        # period). Annual seasonality is instead supplied as exogenous day-of-year Fourier
        # terms, known ahead of time for the whole forecast horizon like the calendar
        # features used elsewhere. The weekly seasonal component uses ARMA terms only
        # (no seasonal differencing): combining seasonal differencing with these exogenous
        # terms destabilized the fit into large oscillations.
        self.order = order
        self.seasonal_period = seasonal_period
        self.annual_harmonics = annual_harmonics
        self.model = None

    def _fourier_terms(self, day_of_year: pd.Series) -> pd.DataFrame:
        values = day_of_year.to_numpy(dtype=float)
        terms = {}
        for k in range(1, self.annual_harmonics + 1):
            terms[f"annual_sin_{k}"] = np.sin(2 * np.pi * k * values / 365.25)
            terms[f"annual_cos_{k}"] = np.cos(2 * np.pi * k * values / 365.25)
        return pd.DataFrame(terms)

    def fit(
        self,
        train: pd.DataFrame,
        target: str,
        features: list[str],
    ):
        exog = self._fourier_terms(train["day_of_year"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = SARIMAX(
                train[target].astype(float),
                exog=exog,
                order=self.order,
                seasonal_order=(2, 0, 1, self.seasonal_period),
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(
                disp=False,
                maxiter=300,
            )

        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")

        exog = self._fourier_terms(future["day_of_year"])
        return np.asarray(self.model.forecast(len(future), exog=exog))
