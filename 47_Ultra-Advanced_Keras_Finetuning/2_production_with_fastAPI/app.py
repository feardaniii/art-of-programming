# ==========================================
# FILE: app.py
# The FastAPI application serving predictions
# ==========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
import json
from typing import List
import time

# ==========================================
# INITIALIZE FASTAPI APP
# ==========================================

app = FastAPI(
    title="Health Prediction API",
    description="Diabetes risk prediction using deep learning",
    version="1.0.0"
)

# ==========================================
# LOAD MODEL & PREPROCESSING AT STARTUP
# ==========================================

print("Loading production model...")
model = keras.models.load_model('health_predictor_production.keras')
scaler = joblib.load('health_scaler.pkl')

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"✅ Model loaded: {metadata['model_name']} v{metadata['version']}")
print(f"   Features: {len(metadata['input_features'])}")
print(f"   Expected inference: {metadata['inference_time_ms']:.2f}ms")

# ==========================================
# REQUEST/RESPONSE MODELS (PYDANTIC)
# ==========================================

class PredictionRequest(BaseModel):
    """
    Patient data for diabetes risk prediction.
    All features are normalized numeric values.
    """
    age: float = Field(..., description="Age (normalized)")
    sex: float = Field(..., description="Sex (normalized)")
    bmi: float = Field(..., description="Body Mass Index (normalized)")
    bp: float = Field(..., description="Average Blood Pressure (normalized)")
    s1: float = Field(..., description="Total Serum Cholesterol")
    s2: float = Field(..., description="Low-Density Lipoproteins")
    s3: float = Field(..., description="High-Density Lipoproteins")
    s4: float = Field(..., description="Total Cholesterol / HDL")
    s5: float = Field(..., description="Log of Serum Triglycerides")
    s6: float = Field(..., description="Blood Sugar Level")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,
                "s1": -0.04,
                "s2": -0.03,
                "s3": -0.00,
                "s4": -0.03,
                "s5": 0.01,
                "s6": -0.02
            }
        }

class PredictionResponse(BaseModel):
    """API response with prediction and metadata."""
    risk_probability: float = Field(..., description="Probability of high diabetes risk (0-1)")
    risk_level: str = Field(..., description="Risk category: 'Low' or 'High'")
    confidence: float = Field(..., description="Model confidence")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used")

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": metadata['model_name'],
        "version": metadata['version'],
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

# ==========================================
# MODEL METRICS ENDPOINT
# ==========================================

@app.get("/metrics")
def get_metrics():
    """Return model performance metrics"""
    return {
        "model_name": metadata['model_name'],
        "version": metadata['version'],
        "performance": metadata['metrics'],
        "inference_time_ms": metadata['inference_time_ms']
    }

# ==========================================
# SINGLE PREDICTION ENDPOINT
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict diabetes risk for a single patient.

    Returns probability and risk level.
    """
    try:
        # Convert request to array
        features = np.array([[
            request.age, request.sex, request.bmi, request.bp,
            request.s1, request.s2, request.s3, request.s4,
            request.s5, request.s6
        ]])

        # Preprocess (scale)
        features_scaled = scaler.transform(features)

        # Predict with timing
        start = time.time()
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        inference_time = (time.time() - start) * 1000

        # Determine risk level
        threshold = metadata['threshold']
        risk_level = "High" if prediction >= threshold else "Low"

        # Confidence: distance from threshold
        confidence = abs(prediction - threshold) / threshold
        confidence = min(confidence, 1.0)

        return PredictionResponse(
            risk_probability=float(prediction),
            risk_level=risk_level,
            confidence=float(confidence),
            inference_time_ms=float(inference_time),
            model_version=metadata['version']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==========================================
# BATCH PREDICTION ENDPOINT
# ==========================================

class BatchRequest(BaseModel):
    """Multiple patient predictions"""
    patients: List[PredictionRequest]

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """
    Predict diabetes risk for multiple patients.
    More efficient than individual requests.
    """
    try:
        # Convert all requests to array
        features_list = []
        for patient in request.patients:
            features_list.append([
                patient.age, patient.sex, patient.bmi, patient.bp,
                patient.s1, patient.s2, patient.s3, patient.s4,
                patient.s5, patient.s6
            ])

        features = np.array(features_list)
        features_scaled = scaler.transform(features)

        # Batch prediction
        start = time.time()
        predictions = model.predict(features_scaled, verbose=0)
        inference_time = (time.time() - start) * 1000

        # Process results
        results = []
        threshold = metadata['threshold']

        for pred in predictions:
            prob = float(pred[0])
            risk_level = "High" if prob >= threshold else "Low"
            confidence = abs(prob - threshold) / threshold

            results.append({
                "risk_probability": prob,
                "risk_level": risk_level,
                "confidence": min(float(confidence), 1.0)
            })

        return {
            "predictions": results,
            "batch_size": len(results),
            "total_inference_time_ms": float(inference_time),
            "avg_inference_time_ms": float(inference_time / len(results)),
            "model_version": metadata['version']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ==========================================
# RUN WITH: uvicorn app:app --reload
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)