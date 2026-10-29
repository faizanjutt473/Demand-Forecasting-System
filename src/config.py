"""Central configuration for the forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_CSV = DATA_DIR / "train.csv"
HOLIDAYS_CSV = DATA_DIR / "holidays_events.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_META_PATH = MODELS_DIR / "best_model_meta.json"

# Forecast horizon (days). Lags are chosen >= HORIZON so the ML model
# never needs its own predictions as inputs (no leakage, no recursion).
HORIZON = 14
N_SPLITS = 5            # number of time-based CV folds
START_DATE = "2015-01-01"  # early 2013-14 data is a ramp-up period, so it is dropped
RANDOM_STATE = 42
