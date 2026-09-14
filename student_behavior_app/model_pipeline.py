import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def generate_student_data(n_samples=1500):
    np.random.seed(42)
    data = {
        'attendance_rate': np.random.uniform(40, 100, n_samples),
        'study_hours_per_week': np.random.uniform(1, 35, n_samples),
        'lms_portal_visits': np.random.randint(2, 120, n_samples),
        'assignment_submission_rate': np.random.uniform(30, 100, n_samples),
        'past_gpa': np.random.uniform(1.0, 4.0, n_samples),
        'participation_score': np.random.uniform(1, 10, n_samples)
    }
    df = pd.DataFrame(data)

    # Target rule with noise: 0 = At-Risk, 1 = On-Track
    score = (
        0.30 * df['attendance_rate'] + 
        0.25 * (df['study_hours_per_week'] / 35 * 100) + 
        0.25 * df['assignment_submission_rate'] + 
        0.20 * (df['past_gpa'] / 4.0 * 100)
    )
    noise = np.random.normal(0, 5, n_samples)
    df['target'] = np.where((score + noise) >= 62, 1, 0)
    return df

def train_and_save():
    print("Generating dataset...")
    df = generate_student_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=['At-Risk', 'On-Track']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModel and Scaler successfully saved to 'models/' directory.")

if __name__ == '__main__':
    train_and_save()