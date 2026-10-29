"""Run CV for all models, pick the winner, retrain it on all data and save it.

Usage (from project root):  python -m src.train
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (FIGURES_DIR, HORIZON, MODEL_META_PATH, MODEL_PATH,
                        MODELS_DIR, N_SPLITS, REPORTS_DIR)
from src.data import load_daily_sales, load_national_holidays
from src.evaluate import time_series_cv
from src.models import LightGBMForecaster, SarimaxForecaster, SeasonalNaive


def plot_series(y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    y.plot(ax=ax)
    ax.set_title("Total daily sales")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "daily_sales.png", dpi=150)
    plt.close(fig)


def plot_last_fold(preds: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    first = next(iter(preds.values()))
    ax.plot(first.index, first["actual"], label="Actual", color="black", linewidth=2)
    for name, df in preds.items():
        ax.plot(df.index, df["forecast"], label=name, marker="o", markersize=3)
    ax.set_title(f"Last CV fold: {HORIZON}-day forecast vs actual")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "last_fold_forecast.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    y = load_daily_sales()
    holidays = load_national_holidays()
    print(f"Series: {y.index.min().date()} -> {y.index.max().date()} ({len(y)} days)")
    plot_series(y)

    factories = {
        "Seasonal Naive": lambda: SeasonalNaive(season=7),
        "SARIMAX": lambda: SarimaxForecaster(),
        "LightGBM": lambda: LightGBMForecaster(holidays),
    }

    fold_tables, summary_rows, last_preds = [], [], {}
    for name, factory in factories.items():
        print(f"Cross-validating {name} ...")
        folds, last_pred = time_series_cv(factory, y, N_SPLITS, HORIZON)
        folds.insert(0, "model", name)
        fold_tables.append(folds)
        last_preds[name] = last_pred
        summary_rows.append({"model": name, "MAE": folds["MAE"].mean(), "RMSE": folds["RMSE"].mean()})

    folds_df = pd.concat(fold_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("RMSE").reset_index(drop=True)
    folds_df.to_csv(REPORTS_DIR / "cv_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "cv_summary.csv", index=False)
    plot_last_fold(last_preds)

    print("\nMean CV results (lower is better):")
    print(summary.round(2).to_string(index=False))

    best_name = summary.loc[0, "model"]
    print(f"\nWinner: {best_name}. Retraining on the full history ...")
    best_model = factories[best_name]().fit(y)
    joblib.dump(best_model, MODEL_PATH)

    meta = {
        "model": best_name,
        "horizon": HORIZON,
        "trained_until": str(y.index[-1].date()),
        "cv_mae": float(summary.loc[0, "MAE"]),
        "cv_rmse": float(summary.loc[0, "RMSE"]),
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
