"""FastAPI service for the best forecasting model.

Run from project root:  uvicorn api.main:app --reload
Docs:                   http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import HORIZON, MODEL_META_PATH, MODEL_PATH

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run `python -m src.train` first.")
    state["model"] = joblib.load(MODEL_PATH)
    state["meta"] = json.loads(MODEL_META_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Sales Forecasting API", version="1.0.0", lifespan=lifespan)


class ForecastRequest(BaseModel):
    horizon: int = Field(7, ge=1, le=HORIZON, description=f"Days to forecast (1-{HORIZON})")


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    model: str
    trained_until: str
    forecasts: list[ForecastPoint]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return state["meta"]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    try:
        pred = state["model"].predict(req.horizon)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
    return ForecastResponse(
        model=state["meta"]["model"],
        trained_until=state["meta"]["trained_until"],
        forecasts=[ForecastPoint(date=d.strftime("%Y-%m-%d"), forecast=round(float(v), 2)) for d, v in pred.items()],
    )
