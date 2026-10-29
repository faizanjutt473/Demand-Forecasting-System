# Report: Retail Demand Forecasting

## 1. Problem and data
## 4. Results (mean over 5 time-based CV folds, 14-day horizon)

| Model | MAE | RMSE |
|---|---|---|
| **LightGBM** | **42,064** | **58,547** |
| SARIMAX | 70,145 | 87,461 |
| Seasonal Naive | 98,696 | 128,217 |

LightGBM reduces MAE by 57% and RMSE by 54% compared with the Seasonal Naive
baseline, and by 40% (MAE) and 33% (RMSE) compared with SARIMAX.
Figures: `figures/daily_sales.png`, `figures/last_fold_forecast.png`.

## 5. Why LightGBM won
- The baseline and SARIMAX only model the weekly pattern and the trend of the series.
  They cannot react to one-off events such as national holidays or month-end/payday effects.
- LightGBM sees these effects directly through the holiday and payday flags, and it also uses
  lag features (14, 21, 28, 35 and 364 days) and rolling mean/std of recent sales,
  so it captures both the level of recent sales and the yearly pattern.
- All lag and rolling features are at least 14 days old, so the comparison is fair and
  there is no data leakage. The same time-based folds were used for all three models.
- SARIMAX still clearly beats the naive baseline, which shows that the weekly seasonality
  is strong, but it is not enough on its own.
- Because LightGBM was best on MAE and RMSE, it was retrained on the full history and
  deployed through the API.

## 2. Approach
- **Seasonal Naive:** sales equal the same weekday of last week.
- **SARIMAX(1,1,1)(1,1,1,7):** captures trend and weekly seasonality.
- **LightGBM:** lags (14, 21, 28, 35, 364), rolling mean/std (shifted by 14 days),
  calendar features, national holidays, payday flag.

## 3. Validation
Expanding-window `TimeSeriesSplit`, 5 folds, 14-day test window. No random split.

## 4. Results (mean over folds)
Paste the table from `reports/cv_summary.csv`:

| Model | MAE | RMSE |
|---|---|---|
| Seasonal Naive | | |
| SARIMAX | | |
| LightGBM | | |

Figures: `figures/daily_sales.png`, `figures/last_fold_forecast.png`.

## 5. Why the winner won
Write 4-5 lines here after you see the results (e.g. did lag/holiday features help,
does the baseline already capture weekly seasonality, where do errors grow?).

## 6. Deployment
The best model is retrained on all data, saved with joblib and served by FastAPI
(`POST /forecast`).

## 7. Limitations and next steps
- Only the total series is forecast, not per store or product family.
- Future promotions and oil price are not used as features.
- Possible improvements: hyperparameter tuning, Prophet, per-store models.
