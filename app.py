import streamlit as st
import numpy as np
import pickle
from data_loader import load_data
from model import train_model
from rag import RAGSystem

st.set_page_config(page_title="AI Heart Health Predictor", layout="centered")

st.title("❤️ AI Health Risk Predictor")
st.warning("⚠️ This is not medical advice. Consult a doctor.")

# Load and train model
df = load_data()
model, scaler = train_model(df)

# Input fields
st.subheader("Enter Health Details")

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.slider("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar >120 (1=True, 0=False)", [1, 0])
restecg = st.slider("Rest ECG (0-2)", 0, 2)
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Angina (1=Yes,0=No)", [1, 0])
oldpeak = st.number_input("Oldpeak")
slope = st.slider("Slope (0-2)", 0, 2)
ca = st.slider("CA (0-3)", 0, 3)
thal = st.slider("Thal (0-3)", 0, 3)

if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    # 🔥 EXPLANATION LOGIC
    reasons = []

    if chol > 240:
        reasons.append("High cholesterol")

    if trestbps > 140:
        reasons.append("High blood pressure")

    if thalach < 100:
        reasons.append("Low maximum heart rate")

    if oldpeak > 2:
        reasons.append("High ST depression (possible heart stress)")

    if exang == 1:
        reasons.append("Exercise-induced angina")

    if age > 55:
        reasons.append("Higher age risk factor")

    # --------------------------
    # OUTPUT
    # --------------------------
    if prediction == 1:
        st.error(f"⚠️ High Risk ({prob*100:.2f}%)")

        if reasons:
            st.write("### 🔍 Reasons:")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("No strong specific risk factors detected, but model predicts high risk.")

    else:
        st.success(f"✅ Low Risk ({prob*100:.2f}%)")

        if reasons:
            st.write("### ⚠️ Minor Risk Factors Present:")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("No significant risk factors detected.")

# RAG Chatbot
st.subheader("💬 Heart Health Assistant")

rag = RAGSystem()
user_query = st.text_input("Ask something about heart health:")

if user_query:
    answer = rag.query(user_query)
    st.write(answer)