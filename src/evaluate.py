"""Metrics and time-based cross-validation."""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def time_series_cv(model_factory: Callable, y: pd.Series, n_splits: int, horizon: int):
    """Expanding-window CV: always train on the past, test on the next ``horizon`` days.

    Returns (per-fold metrics DataFrame, predictions of the last fold).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
    rows, last_fold_pred = [], None

    for fold, (train_idx, test_idx) in enumerate(splitter.split(y), start=1):
        model = model_factory().fit(y.iloc[train_idx])
        pred = model.predict(len(test_idx))
        actual = y.iloc[test_idx]
        rows.append(
            {
                "fold": fold,
                "train_end": y.index[train_idx[-1]].date(),
                "test_start": y.index[test_idx[0]].date(),
                "test_end": y.index[test_idx[-1]].date(),
                "MAE": mae(actual.to_numpy(), pred.to_numpy()),
                "RMSE": rmse(actual.to_numpy(), pred.to_numpy()),
            }
        )
        last_fold_pred = pd.DataFrame({"actual": actual.to_numpy(), "forecast": pred.to_numpy()}, index=actual.index)

    return pd.DataFrame(rows), last_fold_pred
