from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"

app = FastAPI(title="Student Performance Prediction API", version="1.0.0")
model = None


class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0)
    attendance: float = Field(..., ge=0, le=100)
    previous_marks: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float


def load_model_from_disk():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "model.pkl not found. Run the training pipeline before starting the API."
        )
    return joblib.load(MODEL_PATH)


def ensure_model_loaded() -> None:
    global model
    if model is None:
        model = load_model_from_disk()


@app.on_event("startup")
def load_model() -> None:
    ensure_model_loaded()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Student Performance Prediction API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentInput) -> PredictionResponse:
    ensure_model_loaded()

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    frame = pd.DataFrame([
        {
            "study_hours": payload.study_hours,
            "attendance": payload.attendance,
            "previous_marks": payload.previous_marks,
        }
    ])

    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    label = "Pass" if prediction == 1 else "Fail"
    return PredictionResponse(prediction=label, probability=round(probability, 4))
