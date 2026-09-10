"""XGBoost forecaster."""
from ..models import SklearnForecaster


class XGBoostForecaster(SklearnForecaster):
    def __init__(self, seed: int = 42) -> None:
        try: from xgboost import XGBRegressor
        except ImportError as error: raise ImportError("XGBoost requires `pip install -e .[boosting]`") from error
        super().__init__(XGBRegressor(n_estimators=500, max_depth=6, learning_rate=.05, random_state=seed, n_jobs=-1))
