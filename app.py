import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Student Risk Prediction System",
    page_icon="🎓",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

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
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                res = response.json()
                
                # Display outcome card
                if res['is_at_risk']:
                    st.error(f"⚠️ Status: {res['status']} ({res['risk_probability']}% Risk Probability)")
                else:
                    st.success(f"✅ Status: {res['status']} ({res['on_track_probability']}% Confidence)")
                
                # Gauge Chart
                fig = px.pie(
                    names=['Risk Probability', 'On-Track Probability'],
                    values=[res['risk_probability'], res['on_track_probability']],
                    color_discrete_sequence=['#ef553b', '#00cc96'],
                    hole=0.5,
                    title="Risk Assessment Ratio"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interventions
                st.markdown("### 🛠️ Recommended Interventions:")
                for rec in res['recommended_interventions']:
                    st.write(f"- {rec}")
            else:
                st.error("API returned an error. Ensure FastAPI backend is running.")
        except Exception as e:
            st.error(f"Could not connect to FastAPI server. Error: {e}")

st.divider()

# Model Feature Importance Section
st.subheader("📌 Key Risk Factors Driving Predictions")
try:
    model = joblib.load('models/model.pkl')
    features = ['Attendance', 'Study Hours', 'LMS Visits', 'Assignment Rate', 'Past GPA', 'Participation']
    importances = model.feature_importances_
    
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Model Feature Importance Breakdown")
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception:
    st.info("Train the model via `model_pipeline.py` to view feature importances.")