
#  Heart Disease Prediction API

A Machine Learning-based API built with FastAPI to predict the risk of heart disease based on patient health data. The model is trained using the Random Forest Classifier on the UCI Heart Disease dataset.

---

##  Project Structure

```
AIML_TASK/
├── api/
│   ├── main.py              # FastAPI application
│   └── Heart_model.pkl      # Trained Random Forest model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🔍 API Endpoints

| Method | Endpoint       | Description                           |
|--------|----------------|---------------------------------------|
| GET    | `/`            | Welcome message                       |
| GET    | `/health`      | Returns API status                    |
| POST   | `/predict`     | Predicts heart disease risk           |

---

## Input Schema for `/predict`

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Field Descriptions

| Feature    | Type   | Description |
|------------|--------|-------------|
| `age`      | float  | Age of the patient |
| `sex`      | int    | Gender (1 = male, 0 = female) |
| `cp`       | int    | Chest pain type (0–3) |
| `trestbps` | float  | Resting blood pressure (mm Hg) |
| `chol`     | float  | Serum cholesterol (mg/dl) |
| `fbs`      | int    | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg`  | int    | Resting ECG results (0–2) |
| `thalach`  | float  | Maximum heart rate achieved |
| `exang`    | int    | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak`  | float  | ST depression induced by exercise |
| `slope`    | int    | Slope of the peak exercise ST segment (0–2) |
| `ca`       | int    | Number of major vessels colored by fluoroscopy (0–3) |
| `thal`     | int    | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |

---

##  Response from `/predict`

```json
{
  "prediction": "High Risk",
  "probability": 0.82
}
```

- `prediction`: "High Risk" or "Low Risk"
- `probability`: Model's confidence score (0 to 1)

---

##  Model Used

**Model:** Random Forest Classifier  
**Justification:**  
- Handles both numerical and categorical features  
- Works well on imbalanced datasets  
- Reduces overfitting via ensembling  
- Provides feature importance insights  

---

##  Model Performance

| Metric     | Score |
|------------|-------|
| Accuracy   | 86%   |
| Precision  | 85%   |
| Recall     | 87%   |
| F1-Score   | 0.86  |
| ROC-AUC    | 0.91  |

The Random Forest model showed better generalization compared to Logistic Regression and Decision Tree classifiers, making it a strong candidate for this use case.

---

##  Installation & Local Run

```bash
# Clone the repo
git clone https://github.com/sateeshsingamsetti/AIML_TASK.git
cd AIML_TASK

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

##  Test the API

- Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Sample `curl`:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}'
```

---

##  Health Check

```http
GET /health
```

---

##  Deployment

This application is containerized using Docker and ready to be deployed on platforms like **Render**. Dockerfile should point to `api/main.py`.

---

##  License

This project is intended for educational and demonstration purposes only.
