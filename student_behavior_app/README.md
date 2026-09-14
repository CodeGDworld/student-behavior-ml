# Create README.md file
@"
# 🎓 Student Early Warning & Risk Prediction System

An end-to-end Machine Learning web application that predicts student academic risk based on behavioral metrics (attendance, study hours, LMS activity, assignment submission rates).

🚀 **Live Demo:** [Click here to view the live app](https://student-behavior-ml-anantgupta108.streamlit.app)

---

## 🛠️ Features
- **Real-Time Predictions:** Classifies students as **At-Risk** or **On-Track**.
- **Diagnostic Insights:** Highlights primary behavioral drivers affecting performance.
- **Automated Interventions:** Suggests targeted action items for educators and advisors.

---

## 💻 Tech Stack
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Frontend Dashboard:** Streamlit & Plotly
- **Backend API:** FastAPI & Uvicorn

---

## 🚀 Local Setup
```bash
# 1. Clone repository
git clone [https://github.com/CodeGDworld/student-behavior-ml.git](https://github.com/CodeGDworld/student-behavior-ml.git)

# 2. Navigate to project folder
cd student-behavior-ml/student_behavior_app

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run dashboard
python -m streamlit run app.py