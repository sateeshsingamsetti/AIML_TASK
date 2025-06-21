#Fast api
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import path

app = FastAPI()
#model = joblib.load("D:/heart-disease-ml-api/api/Heart_model.pkl")
#model = joblib.load("api\Heart_model.pkl")
model_path = Path(__file__).parent / "Heart_model.pkl"
model = joblib.load(model_path)

class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    chol: float
    thalach: float
    # ... include all required fields

@app.post("/predict")
def predict(data: Patient):
    input_data = np.array([[data.age, data.sex, data.cp, data.chol, data.thalach]])  # Add all fields
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return {"prediction": int(prediction[0]), "probability": float(probability)}

@app.get("/health")
def health():
    return {"status": "API running"}
