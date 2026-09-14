import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Student Risk Prediction System",
    page_icon="🎓",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# Fallback in-memory training function for cloud deployment
@st.cache_resource
def load_or_train_model():
    if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    
    # Train inline if files missing (Cloud environment fallback)
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({
        'attendance_rate': np.random.uniform(40, 100, n),
        'study_hours_per_week': np.random.uniform(1, 35, n),
        'lms_portal_visits': np.random.randint(2, 120, n),
        'assignment_submission_rate': np.random.uniform(30, 100, n),
        'past_gpa': np.random.uniform(1.0, 4.0, n),
        'participation_score': np.random.uniform(1, 10, n)
    })
    score = (0.30 * df['attendance_rate'] + 0.25 * (df['study_hours_per_week'] / 35 * 100) + 
             0.25 * df['assignment_submission_rate'] + 0.20 * (df['past_gpa'] / 4.0 * 100))
    df['target'] = np.where((score + np.random.normal(0, 5, n)) >= 62, 1, 0)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def generate_recommendations(features: dict, risk_prob: float) -> list:
    recs = []
    if risk_prob > 0.5:
        recs.append("High Risk Alert: Flag for immediate academic advisor intervention.")
    if features['attendance_rate'] < 75.0:
        recs.append("Schedule attendance counseling session.")
    if features['assignment_submission_rate'] < 70.0:
        recs.append("Assign tutoring support for pending course assignments.")
    if features['study_hours_per_week'] < 8.0:
        recs.append("Recommend peer study group enrollment.")
    if not recs:
        recs.append("Student is performing well. Maintain standard monitoring.")
    return recs

st.title("🎓 Student Early Warning & Behavior Prediction System")
st.markdown("Predict student risk status early in the semester to enable timely academic interventions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Enter Student Behavioral Metrics")
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0)
    study_hours = st.slider("Weekly Study Hours", 0.0, 40.0, 12.0)
    lms_visits = st.number_input("LMS Portal Logins / Activity", 0, 200, 35)
    assignment_rate = st.slider("Assignment Submission Rate (%)", 0.0, 100.0, 80.0)
    past_gpa = st.number_input("Past Cumulative GPA", 0.0, 4.0, 2.8, step=0.1)
    participation = st.slider("Class Participation Score (1-10)", 1.0, 10.0, 5.0)
    submit = st.button("Analyze Student Risk", type="primary")

with col2:
    st.subheader("📊 Diagnostic Report & Recommendations")
    if submit:
        payload = {
            "attendance_rate": attendance,
            "study_hours_per_week": study_hours,
            "lms_portal_visits": lms_visits,
            "assignment_submission_rate": assignment_rate,
            "past_gpa": past_gpa,
            "participation_score": participation
        }
        
        # Try local API first; fall back to in-memory model if API fails
        res = None
        try:
            response = requests.post(API_URL, json=payload, timeout=2)
            if response.status_code == 200:
                res = response.json()
        except Exception:
            # Fallback to direct model inference
            model, scaler = load_or_train_model()
            df_input = pd.DataFrame([payload])
            scaled_input = scaler.transform(df_input)
            pred = int(model.predict(scaled_input)[0])
            probs = model.predict_proba(scaled_input)[0]
            risk_prob = float(probs[0])
            
            res = {
                "status": "At-Risk" if pred == 0 else "On-Track",
                "is_at_risk": pred == 0,
                "risk_probability": round(risk_prob * 100, 2),
                "on_track_probability": round(probs[1] * 100, 2),
                "recommended_interventions": generate_recommendations(payload, risk_prob)
            }
        
        if res:
            if res['is_at_risk']:
                st.error(f"⚠️ Status: {res['status']} ({res['risk_probability']}% Risk Probability)")
            else:
                st.success(f"✅ Status: {res['status']} ({res['on_track_probability']}% Confidence)")
            
            fig = px.pie(
                names=['Risk Probability', 'On-Track Probability'],
                values=[res['risk_probability'], res['on_track_probability']],
                color_discrete_sequence=['#ef553b', '#00cc96'],
                hole=0.5,
                title="Risk Assessment Ratio"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🛠️ Recommended Interventions:")
            for rec in res['recommended_interventions']:
                st.write(f"- {rec}")

st.divider()

st.subheader("📌 Key Risk Factors Driving Predictions")
try:
    model, _ = load_or_train_model()
    features = ['Attendance', 'Study Hours', 'LMS Visits', 'Assignment Rate', 'Past GPA', 'Participation']
    importances = model.feature_importances_
    
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Model Feature Importance Breakdown")
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception:
    st.info("Train the model to view feature importances.")