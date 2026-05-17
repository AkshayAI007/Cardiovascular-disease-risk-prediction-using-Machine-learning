"""
FastAPI REST endpoint for cardiovascular risk prediction.

Start:
    uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload

Test (browser):
    http://localhost:8000/docs          ← interactive Swagger UI

Test (curl):
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"age":55,"sex":1,"is_smoking":0,"cigsPerDay":0,"BPMeds":0,
              "prevalentStroke":0,"prevalentHyp":1,"diabetes":0,
              "totChol":250,"sysBP":140,"diaBP":90,"BMI":28.5,
              "heartRate":75,"glucose":100,"education":2}'
"""
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_inference

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_bundle.joblib")
UI_PATH = os.path.join(os.path.dirname(__file__), "Cardiovascular Risk Assessment.html")

app = FastAPI(
    title="Cardiovascular Risk Predictor",
    description="Predicts 10-year coronary heart disease risk from clinical features.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(BUNDLE_PATH):
            raise RuntimeError("Model bundle not found. Run `python src/train.py` first.")
        _bundle = joblib.load(BUNDLE_PATH)
    return _bundle


class PatientInput(BaseModel):
    age:             float = Field(..., ge=20, le=100, json_schema_extra={"example": 55})
    sex:             int   = Field(..., ge=0, le=1, description="1=Male, 0=Female")

    def to_model_dict(self) -> dict:
        d = self.model_dump()
        d["sex"] = "M" if d["sex"] == 1 else "F"
        return d
    is_smoking:      int   = Field(..., ge=0, le=1)
    cigsPerDay:      float = Field(0, ge=0, le=100)
    BPMeds:          int   = Field(0, ge=0, le=1)
    prevalentStroke: int   = Field(0, ge=0, le=1)
    prevalentHyp:    int   = Field(0, ge=0, le=1)
    diabetes:        int   = Field(0, ge=0, le=1)
    totChol:         float = Field(..., ge=100, le=700, json_schema_extra={"example": 250})
    sysBP:           float = Field(..., ge=80,  le=300, json_schema_extra={"example": 140})
    diaBP:           float = Field(..., ge=40,  le=200, json_schema_extra={"example": 90})
    BMI:             float = Field(..., ge=10,  le=60,  json_schema_extra={"example": 28.5})
    heartRate:       float = Field(..., ge=30,  le=200, json_schema_extra={"example": 75})
    glucose:         float = Field(..., ge=40,  le=500, json_schema_extra={"example": 100})
    education:       float = Field(2,   ge=1,   le=4)


class PredictionResponse(BaseModel):
    model_name:      str
    threshold:       float
    chd_probability: float
    chd_prediction:  int
    risk_label:      str


@app.get("/")
def ui():
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "model": get_bundle()["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    bundle = get_bundle()

    df = pd.DataFrame([patient.to_model_dict()])
    artifacts = {
        "fill_values":  bundle["fill_values"],
        "stat_params":  bundle["stat_params"],
        "scaler":       bundle["scaler"],
        "feature_cols": bundle["feature_cols"],
    }

    try:
        X = preprocess_inference(df, artifacts)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    threshold = bundle.get("threshold", 0.5)
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    return PredictionResponse(
        model_name=bundle["model_name"],
        threshold=round(threshold, 3),
        chd_probability=round(prob, 4),
        chd_prediction=pred,
        risk_label="High Risk" if pred == 1 else "Low Risk",
    )


@app.post("/predict/batch")
def predict_batch(patients: list[PatientInput]):
    bundle = get_bundle()
    df = pd.DataFrame([p.to_model_dict() for p in patients])
    artifacts = {
        "fill_values":  bundle["fill_values"],
        "stat_params":  bundle["stat_params"],
        "scaler":       bundle["scaler"],
        "feature_cols": bundle["feature_cols"],
    }
    X = preprocess_inference(df, artifacts)
    threshold = bundle.get("threshold", 0.5)
    probs = bundle["model"].predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return [
        {
            "index":           i,
            "chd_probability": round(float(p), 4),
            "chd_prediction":  int(pred),
            "risk_label":      "High Risk" if pred else "Low Risk",
        }
        for i, (p, pred) in enumerate(zip(probs, preds))
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
