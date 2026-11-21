import json
import pandas as pd
import joblib
from src.explain import ChurnExplainer

# Load pipeline and threshold
pipeline = joblib.load('models/churn_pipeline.joblib')
with open('models/threshold.json') as f:
    thresh_data = json.load(f)
threshold = thresh_data.get('optimal_threshold', thresh_data.get('threshold', 0.5))

explainer = ChurnExplainer()

cases = {
    "Case 1: High Risk (Senior, Month-to-Month, Fiber Optic, No Support)": {
        "gender": "Female",
        "SeniorCitizen": "1",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00
    },
    "Case 2: Low Risk / Loyal (65 Months, 2-Year Contract, Full Bundled Protection)": {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.00,
        "TotalCharges": 2925.00
    },
    "Case 3: Borderline (16 Months, 1-Year Contract, Fiber Optic, Partial Services)": {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 16,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 75.00,
        "TotalCharges": 1200.00
    }
}

for name, payload in cases.items():
    df = pd.DataFrame([payload])
    prob = float(pipeline.predict_proba(df)[0, 1])
    pred = "Will churn" if prob >= threshold else "Will stay"
    reasons = explainer.get_top_features(payload, k=3)
    print("=" * 60)
    print(f"{name}")
    print("=" * 60)
    print(f"Churn Probability : {prob:.4f} ({prob*100:.1f}%)")
    print(f"Decision Threshold: {threshold:.2f}")
    print(f"Model Prediction  : {pred}")
    print("Top 3 SHAP Drivers:")
    for i, r in enumerate(reasons, 1):
        direction_icon = "▲" if "increase" in r["direction"].lower() else "▼"
        print(f"  {i}. {r['feature']} ({r['value']}) -> {direction_icon} {r['direction']} (Impact: {r['impact']:+.4f})")
    print()
