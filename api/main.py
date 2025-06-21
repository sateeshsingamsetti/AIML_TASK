from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# Load model from the same folder
mmodel_path = Path(__file__).resolve().parent / "Heart_model.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)


# Input schema
class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    chol: float
    thalach: float

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: Patient):
    input_data = np.array([[data.age, data.sex, data.cp, data.chol, data.thalach]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": risk,
        "probability": round(float(probability), 4)
    }

@app.get("/health")
def health():
    return {"status": "API running"}
