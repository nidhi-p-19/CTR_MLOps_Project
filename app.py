# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
from typing import Dict, Any

app = FastAPI(title="CTR Prediction API (LightGBM)", version="1.0")

# --- Load model & feature schema ---
MODEL_PATH = "model/ctr_model_lgb.pkl"
FEATS_PATH = "model/feature_names.pkl"
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/schema: {e}")

class CTRPayload(BaseModel):
    # Flexible: accept any keys; we’ll reindex to feature_names
    features: Dict[str, Any] = Field(..., description="Key-value pairs of model features")

@app.get("/")
def root():
    return {"message": "CTR Prediction API is running 🚀"}

@app.get("/model/info")
def model_info():
    return {
        "model_file": MODEL_PATH,
        "feature_count": len(feature_names),
        "features_preview": feature_names[:10]
    }

@app.post("/predict")
def predict(payload: CTRPayload):
    try:
        X = pd.DataFrame([payload.features])

        # Coerce numeric where possible (strings to numbers)
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="ignore")

        # Reindex to training schema (missing -> 0)
        X = X.reindex(columns=feature_names, fill_value=0)

        pred = float(model.predict(X)[0])
        return {"click_probability": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
