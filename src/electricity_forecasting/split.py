from __future__ import annotations

import pandas as pd


def last_year_train_test_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out exactly the newest calendar year; no validation partition is created."""
    data = frame.sort_values("Date").reset_index(drop=True)
    latest_date = pd.Timestamp(data["Date"].max())
    cutoff = latest_date - pd.DateOffset(years=1)
    train, test = data[data["Date"] <= cutoff].copy(), data[data["Date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("At least one historical year plus the final test year is required")
    return train, test
