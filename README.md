# Retail Demand Forecasting System with Time-Series Models and FastAPI Deployment

Forecasts total daily retail sales (Kaggle: *Store Sales - Time Series Forecasting*)
with three models, compares them using time-based cross-validation, and serves
the winner through a REST API.

## Models
| Model | Type | Idea |
|---|---|---|
| Seasonal Naive | Baseline | Same weekday of last week |
| SARIMAX | Statistical | ARIMA(1,1,1)(1,1,1,7) with weekly seasonality |
| LightGBM | Machine learning | Lag, rolling and calendar/holiday features |

## Validation
- `TimeSeriesSplit` (expanding window, 5 folds, each fold tests the next 14 days). No random split.
- All lag/rolling features are at least 14 days old, so there is no data leakage.
- Metrics: **MAE** and **RMSE**, averaged over folds.

## Project structure
```
.
├── api/main.py          # FastAPI service
├── data/                # Kaggle CSV files (not committed)
├── models/              # saved best model
├── reports/             # cv_summary.csv, cv_folds.csv, figures/, report.md
├── src/
│   ├── config.py        # paths and constants
│   ├── data.py          # loading + daily aggregation
│   ├── features.py      # lag / rolling / calendar features
│   ├── models.py        # the 3 forecasters
│   ├── evaluate.py      # MAE, RMSE, time-based CV
│   └── train.py         # CV, model selection, training, saving
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt

kaggle competitions download -c store-sales-time-series-forecasting -p data --unzip

python -m src.train                # CV + pick winner + save model
uvicorn api.main:app --reload      # start the API
```

## API
Interactive docs: http://127.0.0.1:8000/docs

```bash
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d "{\"horizon\": 7}"
```
Endpoints: `GET /health`, `GET /model-info`, `POST /forecast` (horizon 1-14 days).

## Results
See `reports/cv_summary.csv` and `reports/report.md`.
