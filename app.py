import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("isolation_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# App configuration
st.set_page_config(page_title="Anomaly Detection App", page_icon="🩺", layout="centered")
st.title("💡 Medicare Anomaly Detection")
st.markdown("Enter patient details below to predict potential anomalies in Medicare claims.")

# Input form
with st.form(key='patient_form'):
    id_val = st.number_input("Patient ID", min_value=1, step=1)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    income = st.number_input("Income", min_value=0, step=1000)
    unique_procedures = st.number_input("Unique Procedures", min_value=0, step=1)
    total_procedures_count = st.number_input("Total Procedures Count", min_value=0, step=1)
    total_counts = st.number_input("Total Number of Transactions", min_value=0, step=1)

    submit = st.form_submit_button("Predict Anomaly")

if submit:
    # Prepare input for prediction
    input_data = np.array([[id_val, age, gender, income, unique_procedures, total_procedures_count, total_counts]])
    prediction = model.predict(input_data)
    score = model.decision_function(input_data)[0]

    # Results
    if prediction[0] == -1:
        st.error(f"⚠️ Anomaly Detected!\n\nDecision Score: {score:.5f}")
    else:
        st.success(f"✅ Normal Case.\n\nDecision Score: {score:.5f}")

    st.info("ℹ️ *Lower decision scores may indicate potential anomalies.*")



