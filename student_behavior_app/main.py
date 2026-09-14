import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Risk Prediction API",
    description="Early Warning API for Student Academic At-Risk Detection",
    version="1.0.0"
)

# Load saved artifacts
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    raise RuntimeError("Models not found. Run model_pipeline.py first!") from e

class StudentFeatures(BaseModel):
    attendance_rate: float = Field(..., ge=0, le=100, example=65.5)
    study_hours_per_week: float = Field(..., ge=0, le=100, example=10.0)
    lms_portal_visits: int = Field(..., ge=0, example=25)
    assignment_submission_rate: float = Field(..., ge=0, le=100, example=70.0)
    past_gpa: float = Field(..., ge=0, le=4.0, example=2.4)
    participation_score: float = Field(..., ge=1, le=10, example=4.0)

def generate_recommendations(features: dict, risk_prob: float) -> list:
    recommendations = []
    if risk_prob > 0.5:
        recommendations.append("High Risk Alert: Flag for immediate academic advisor intervention.")
    if features['attendance_rate'] < 75.0:
        recommendations.append("Schedule attendance counseling session.")
    if features['assignment_submission_rate'] < 70.0:
        recommendations.append("Assign tutoring support for pending course assignments.")
    if features['study_hours_per_week'] < 8.0:
        recommendations.append("Recommend peer study group enrollment.")
    if not recommendations:
        recommendations.append("Student is performing well. Maintain standard monitoring.")
    return recommendations

@app.get("/")
def read_root():
    return {"status": "online", "message": "Student Behavior API is active"}

@app.post("/predict")
def predict_student_risk(student: StudentFeatures):
    data = pd.DataFrame([student.model_dump()])
    scaled_data = scaler.transform(data)
    
    prediction = int(model.predict(scaled_data)[0])
    probabilities = model.predict_proba(scaled_data)[0]
    risk_probability = float(probabilities[0]) # Class 0 probability
    
    status = "At-Risk" if prediction == 0 else "On-Track"
    recs = generate_recommendations(student.model_dump(), risk_probability)
    
    return {
        "status": status,
        "is_at_risk": prediction == 0,
        "risk_probability": round(risk_probability * 100, 2),
        "on_track_probability": round(probabilities[1] * 100, 2),
        "recommended_interventions": recs
    }