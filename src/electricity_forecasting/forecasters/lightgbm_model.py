"""LightGBM forecaster."""
from ..models import SklearnForecaster


class LightGBMForecaster(SklearnForecaster):
    def __init__(self, seed: int = 42) -> None:
        try: from lightgbm import LGBMRegressor
        except ImportError as error: raise ImportError("LightGBM requires `pip install -e .[boosting]`") from error
        super().__init__(LGBMRegressor(n_estimators=500, learning_rate=.05, random_state=seed, verbosity=-1))
