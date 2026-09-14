import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Patient Health Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load("health_risk_model.pkl")
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

st.title("🩺 Patient Disease & Clinical Risk Prediction System")
st.markdown(
    "A machine learning diagnostic screening tool designed to evaluate patient vitals "
    "and predict metabolic risk levels."
)

if artifacts is None:
    st.error("Model artifacts not found! Run `python train_model.py` in the terminal first.")
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]

st.sidebar.header("Patient Vitals & Demographics")

age = st.sidebar.slider("Age (years)", min_value=18, max_value=90, value=45, step=1)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=55.0, value=27.4, step=0.1)
glucose = st.sidebar.slider("Fasting Blood Glucose (mg/dL)", min_value=60, max_value=300, value=110, step=1)
hba1c = st.sidebar.number_input("HbA1c Level (%)", min_value=3.5, max_value=15.0, value=5.7, step=0.1)
systolic_bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=210, value=125, step=1)
cholesterol = st.sidebar.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=195, step=1)
hypertension = st.sidebar.selectbox("Diagnosed Hypertension History?", ("No", "Yes"))
heart_disease = st.sidebar.selectbox("History of Cardiovascular Disease?", ("No", "Yes"))

ht_val = 1 if hypertension == "Yes" else 0
hd_val = 1 if heart_disease == "Yes" else 0

input_dict = {
    "Age": age,
    "BMI": bmi,
    "Glucose": glucose,
    "HbA1c": hba1c,
    "SystolicBP": systolic_bp,
    "Cholesterol": cholesterol,
    "Hypertension": ht_val,
    "HeartDisease": hd_val
}
input_df = pd.DataFrame([input_dict])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Patient Record")
    st.dataframe(input_df.T.rename(columns={0: "Entered Value"}))

with col2:
    st.subheader("Diagnostic Assessment")
    
    if st.button("Calculate Risk Assessment", type="primary"):
        scaled_input = scaler.transform(input_df[features])
        prob = model.predict_proba(scaled_input)[0][1] * 100

        if prob < 35:
            st.success(f"**Low Clinical Risk** ({prob:.1f}% estimated risk score)")
            st.info("Patient's metabolic indicators fall within normal clinical baselines.")
        elif 35 <= prob < 65:
            st.warning(f"**Moderate / Borderline Risk** ({prob:.1f}% estimated risk score)")
            st.info("Elevated indicators detected. Recommend lifestyle intervention and regular monitoring.")
        else:
            st.error(f"**High Clinical Risk** ({prob:.1f}% estimated risk score)")
            st.info("High probability of underlying metabolic dysfunction. Recommend diagnostic HbA1c lab tests.")

        st.progress(int(prob))

        st.write("---")
        st.markdown("### Primary Contributing Indicators")
        alerts = []
        if glucose >= 126:
            alerts.append("• **Fasting Glucose ≥ 126 mg/dL:** Meets criteria for diabetic evaluation.")
        if hba1c >= 6.5:
            alerts.append("• **HbA1c ≥ 6.5%:** High chronic glycemic levels.")
        if bmi >= 30:
            alerts.append("• **BMI ≥ 30 (Obese):** Known high-leverage risk factor.")
        if systolic_bp >= 130 or ht_val == 1:
            alerts.append("• **Hypertension Present:** Elevates secondary vascular risk.")

        if alerts:
            for alert in alerts:
                st.write(alert)
        else:
            st.write("No acute single-indicator threshold triggers detected.")