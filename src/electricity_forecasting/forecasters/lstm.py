"""PyTorch LSTM forecaster.

Consumes the same supervised feature set as the tree models (calendar, lag,
rolling, and social columns from `features.make_supervised`) instead of a raw
demand-only window, so it can see day-of-week/seasonal structure and the
annual social indicators directly. Multi-step forecasts are produced the same
way the tree models' recursive forecast is: target-derived lag/rolling
columns are recomputed from true-then-predicted history via
`resolve_recursive_features`, while calendar/social columns (known ahead of
time) are taken as-is from the future frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import resolve_recursive_features
from ..models import Forecaster


class LSTMForecaster(Forecaster):
    def __init__(
        self,
        lookback: int = 28,
        epochs: int = 150,
        hidden_size: int = 64,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seed = seed
        self.network = None
        self.features: list[str] = []
        self.target_mean = self.target_std = None
        self.feature_mean = self.feature_std = None
        self.history: list[float] = []
        self._buffer: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, target: str, features: list[str]):
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise ImportError("LSTM requires `pip install -e .[deep-learning]`") from error

        self.features = list(features)
        usable = train.dropna(subset=self.features + [target]).sort_values("Date").reset_index(drop=True)
        if len(usable) <= self.lookback:
            raise ValueError("Training set is shorter than the LSTM lookback after dropping incomplete feature rows")

        torch.manual_seed(self.seed)
        target_values = usable[target].to_numpy(dtype=np.float32)
        self.target_mean, self.target_std = float(target_values.mean()), float(target_values.std() or 1.0)

        feature_matrix = usable[self.features].to_numpy(dtype=np.float32)
        self.feature_mean = feature_matrix.mean(axis=0)
        self.feature_std = np.where(feature_matrix.std(axis=0) == 0, 1, feature_matrix.std(axis=0))
        scaled_features = (feature_matrix - self.feature_mean) / self.feature_std
        scaled_target = (target_values - self.target_mean) / self.target_std

        starts = range(self.lookback - 1, len(scaled_features))
        x = np.array([scaled_features[i - self.lookback + 1:i + 1] for i in starts], dtype=np.float32)
        y = scaled_target[self.lookback - 1:, None]

        class Network(nn.Module):
            def __init__(self, hidden: int, inputs: int):
                super().__init__()
                self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
                self.output = nn.Linear(hidden, 1)

            def forward(self, inputs):
                return self.output(self.lstm(inputs)[0][:, -1, :])

        self.network = Network(self.hidden_size, x.shape[2])
        optimiser, criterion = torch.optim.Adam(self.network.parameters(), lr=1e-3), nn.MSELoss()
        inputs_tensor, outputs_tensor = torch.tensor(x), torch.tensor(y)
        generator = torch.Generator().manual_seed(self.seed)
        sample_count = len(inputs_tensor)
        self.network.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, self.batch_size):
                batch_index = permutation[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.network(inputs_tensor[batch_index]), outputs_tensor[batch_index])
                loss.backward()
                optimiser.step()
        self.network.eval()

        self.history = list(train[target].to_numpy(dtype=float))
        self._buffer = scaled_features[-self.lookback:].copy()
        return self

    def _scale_row(self, candidate: pd.Series) -> np.ndarray:
        values = candidate[self.features].to_numpy(dtype=np.float32)
        return (values - self.feature_mean) / self.feature_std

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if self.network is None or self._buffer is None:
            raise RuntimeError("Model is not fitted")
        import torch
        history, buffer, forecasts = list(self.history), self._buffer.copy(), []
        with torch.no_grad():
            for _, row in future.iterrows():
                candidate = resolve_recursive_features(row, self.features, history)
                scaled_row = self._scale_row(candidate)
                buffer = np.vstack([buffer[1:], scaled_row[None, :]])
                inputs = torch.tensor(buffer[None, :, :])
                value = float(self.network(inputs).item() * self.target_std + self.target_mean)
                forecasts.append(value)
                history.append(value)
        return np.asarray(forecasts)
