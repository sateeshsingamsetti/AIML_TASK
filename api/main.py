from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# Load model
model_path = Path(__file__).resolve().parent / "Heart_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model = joblib.load(model_path)

# Define full feature model
class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: Patient):
    try:
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        #risk = "High Risk" if prediction == 1 else "Low Risk"
        if probability > 0.70:
            risk = "High Risk"
        elif probability > 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "prediction": risk,
            "probability": round(float(probability), 4)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "API running"}
