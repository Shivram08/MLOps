from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import (
    load_model,
    predict_one,
    predict_proba_one,
    explain_one,
    get_metrics,
    get_model_card,
)

app = FastAPI(title="Penguins Species Classifier API", version="1.0.0")

# Load model once at startup
try:
    MODEL = load_model()
except Exception:
    MODEL = None


class PenguinFeatures(BaseModel):
    flipper_length_mm: float = Field(..., example=190)
    body_mass_g: float = Field(..., example=4200)
    island: str = Field(..., example="Biscoe")
    sex: str = Field(..., example="MALE")


class PredictResponse(BaseModel):
    prediction: str


class PredictProbaResponse(BaseModel):
    probabilities: Dict[str, float]
    prediction: str


class ExplainResponse(BaseModel):
    predicted_class: str
    top_contributions: list
    note: str


@app.get("/")
def root():
    return {"message": "Penguins classifier is running. Visit /docs for API documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model and restart the API.")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PenguinFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model and restart the API.")
    try:
        pred = predict_one(MODEL, payload.model_dump())
        return PredictResponse(prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(payload: PenguinFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model and restart the API.")
    try:
        probs = predict_proba_one(MODEL, payload.model_dump())
        pred = max(probs, key=probs.get)
        return PredictProbaResponse(probabilities=probs, prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/model_info")
def model_info():
    info = get_model_card()
    if not info:
        raise HTTPException(status_code=404, detail="model_card.json not found. Train the model first.")
    return info


@app.post("/explain")
def explain(payload: PenguinFeatures, top_k: int = 5):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model and restart the API.")
    try:
        return explain_one(MODEL, payload.model_dump(), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
