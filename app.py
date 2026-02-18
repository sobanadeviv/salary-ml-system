from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Salary Prediction API")

# Load trained model
model = joblib.load("models/salary_model.pkl")


# Define input schema
class SalaryInput(BaseModel):
    YearsCodePro: float
    Country: str
    EdLevel: str
    Employment: str
    DevType: str
    OrgSize: str


@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}


@app.post("/predict")
def predict_salary(data: SalaryInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict log salary
    log_prediction = model.predict(input_df)[0]

    # Convert back to actual salary
    salary_prediction = float(np.exp(log_prediction))

    return {
        "predicted_salary_usd": round(salary_prediction, 2)
    }
