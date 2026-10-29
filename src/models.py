"""Three forecasters sharing one interface: fit(y) and predict(horizon)."""
import numpy as np
import pandas as pd

from src.config import HORIZON, RANDOM_STATE
from src.features import build_features


class BaseForecaster:
    name = "base"

    def fit(self, y: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _future_index(y: pd.Series, horizon: int) -> pd.DatetimeIndex:
        start = y.index[-1] + pd.Timedelta(days=1)
        return pd.date_range(start, periods=horizon, freq="D")


class SeasonalNaive(BaseForecaster):
    """Baseline: the same weekday of last week."""

    name = "Seasonal Naive"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, y):
        self.y_ = y.copy()
        return self

    def predict(self, horizon):
        last_season = self.y_.iloc[-self.season:].to_numpy()
        values = np.resize(last_season, horizon)  # repeats the last week
        return pd.Series(values, index=self._future_index(self.y_, horizon))


class SarimaxForecaster(BaseForecaster):
    """Statistical model: seasonal ARIMA with weekly seasonality."""

    name = "SARIMAX"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.y_ = y.copy()
        self.result_ = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, horizon):
        values = np.clip(np.asarray(self.result_.forecast(horizon)), 0, None)
        return pd.Series(values, index=self._future_index(self.y_, horizon))


class LightGBMForecaster(BaseForecaster):
    """ML model: LightGBM on lag, rolling and calendar features."""

    name = "LightGBM"

    def __init__(self, holidays: pd.DatetimeIndex, params: dict | None = None):
        self.holidays = holidays
        self.params = params or dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    def fit(self, y):
        from lightgbm import LGBMRegressor

        self.y_ = y.copy()
        data = build_features(y, self.holidays).assign(target=y).dropna()
        self.feature_names_ = [c for c in data.columns if c != "target"]
        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(data[self.feature_names_], data["target"])
        return self

    def predict(self, horizon):
        if horizon > HORIZON:
            raise ValueError(f"horizon must be <= {HORIZON}")
        idx = self._future_index(self.y_, horizon)
        extended = pd.concat([self.y_, pd.Series(np.nan, index=idx)])
        feats = build_features(extended, self.holidays).loc[idx, self.feature_names_]
        values = np.clip(self.model_.predict(feats), 0, None)
        return pd.Series(values, index=idx)
