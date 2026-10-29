"""Feature engineering: calendar + lag + rolling features."""
import pandas as pd

from src.config import HORIZON

# All lags are >= HORIZON, so a feature for day D only uses data
# that is known HORIZON days before D.
LAGS = [HORIZON, 21, 28, 35, 364]
ROLLING_WINDOWS = [7, 28]


def build_features(y: pd.Series, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the feature matrix for every date in ``y``.

    ``y`` may contain NaN for future dates; the features of those dates are
    still valid as long as they are at most HORIZON days after the last
    known value.
    """
    idx = y.index
    feats = pd.DataFrame(index=idx)

    # Calendar features
    feats["dayofweek"] = idx.dayofweek
    feats["day"] = idx.day
    feats["month"] = idx.month
    feats["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()
    feats["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feats["is_payday"] = ((idx.day == 15) | idx.is_month_end).astype(int)
    feats["is_holiday"] = idx.isin(holidays).astype(int)

    # Lag features
    for lag in LAGS:
        feats[f"lag_{lag}"] = y.shift(lag)

    # Rolling statistics (shifted by HORIZON to avoid leakage)
    shifted = y.shift(HORIZON)
    for w in ROLLING_WINDOWS:
        feats[f"roll_mean_{w}"] = shifted.rolling(w).mean()
    feats["roll_std_7"] = shifted.rolling(7).std()

    return feats
