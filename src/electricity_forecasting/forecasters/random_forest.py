"""Random Forest forecaster."""
from sklearn.ensemble import RandomForestRegressor
from ..models import SklearnForecaster


class RandomForestForecaster(SklearnForecaster):
    def __init__(self, seed: int = 42) -> None:
        # max_features="sqrt" decorrelates the trees (sklearn's regressor default of 1.0
        # considers every feature at every split, so the ensemble barely differs from one
        # deep tree and collapses badly once the recursive forecast drifts from training
        # data); min_samples_leaf=5 adds further regularization against that drift.
        super().__init__(RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt", random_state=seed, n_jobs=-1
        ))
