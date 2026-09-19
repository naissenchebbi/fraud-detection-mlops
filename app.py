from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Charger le modèle et le scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire",
    version="1.0.0"
)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convertir en array
        data = np.array([list(transaction.dict().values())])
        data_scaled = scaler.transform(data)
        
        # Prédiction
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "status": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ Legitimate transaction"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))