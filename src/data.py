"""Data loading and preparation."""
import pandas as pd

from src.config import HOLIDAYS_CSV, START_DATE, TRAIN_CSV


def load_daily_sales() -> pd.Series:
    """Total sales of all stores and product families per day.

    Days missing from the file (e.g. 25 December, stores closed) are
    inserted with 0 sales so the series has a continuous daily index.
    """
    df = pd.read_csv(TRAIN_CSV, usecols=["date", "sales"], parse_dates=["date"])
    daily = df.groupby("date")["sales"].sum().asfreq("D").fillna(0.0)
    daily = daily[daily.index >= START_DATE]
    daily.name = "sales"
    return daily


def load_national_holidays() -> pd.DatetimeIndex:
    """Dates of national holidays that were actually celebrated."""
    hol = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])
    mask = (
        (hol["locale"] == "National")
        & (~hol["transferred"].astype(bool))
        & (hol["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]))
    )
    return pd.DatetimeIndex(sorted(hol.loc[mask, "date"].unique()))
