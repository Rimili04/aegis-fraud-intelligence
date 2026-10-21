from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import numpy as np

# Dynamic sys path inclusion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.src.features import build_features
from ml.src.validation import validate_payload

app = FastAPI(title="PROJECT AEGIS — Adaptive Enterprise Guard & Intelligence System", version="1.0.0")

# Lazy model loader
MODEL_PATH = 'aegis-fraud-intelligence/ml/artifacts/model.joblib'
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class SingleTransactionPayload(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "engine": "Project AEGIS Core"
    }

@app.post("/api/v1/predict/single")
def predict_single(payload: SingleTransactionPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available.")
    
    # Prepare record dataframe
    data = payload.dict()
    df = pd.DataFrame([data])
    
    # Validate payload features
    anomalies = validate_payload(df)
    
    # Compute feature vector
    feats = build_features(df)
    
    # Run inference pipeline
    prob = model.predict_proba(feats)[:, 1][0]
    score = round(prob * 100, 1)
    
    decision_category = "LOW"
    if score >= 75.0:
        decision_category = "HIGH"
    elif score >= 40.0:
        decision_category = "MEDIUM"
        
    return {
        "risk_score": score,
        "risk_category": decision_category,
        "anomalies_detected": anomalies,
        "model_version": "v1.0-paysim"
    }

@app.post("/api/v1/predict/batch")
def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    try:
        df_batch = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV: {e}")
        
    # Schema check
    anomalies = validate_payload(df_batch)
    features = build_features(df_batch)
    
    # Prediction
    probs = model.predict_proba(features)[:, 1]
    scores = np.round(probs * 100, 1)
    
    # Output assembly
    df_batch['risk_score'] = scores
    df_batch['risk_category'] = np.select(
        [scores >= 75.0, (scores >= 40.0) & (scores < 75.0), scores < 40.0],
        ['HIGH', 'MEDIUM', 'LOW'],
        default='LOW'
    )
    
    results = df_batch[['step', 'type', 'amount', 'risk_score', 'risk_category']].head(1000).to_dict(orient='records')
    return {
        "total_processed": len(df_batch),
        "anomalies": anomalies,
        "preview_results": results
    }
