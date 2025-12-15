import os
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}
    .stButton>button {width: 100%; padding: 0.7rem; border-radius: 10px;}
    .card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("artifacts/heart_mlp_tf.keras")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_artifacts()

LOG_FILE = "predictions_log.csv"

def append_log(row_dict: dict, log_path: str = LOG_FILE) -> None:
    row = pd.DataFrame([row_dict])
    if os.path.exists(log_path):
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, mode="w", header=True, index=False)


st.title("Heart Disease Prediction")
st.write("")


with st.form("input_form"):
    st.subheader("Patient Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col3:
        dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

    st.subheader("Clinical Measurements")

    col1, col2, col3 = st.columns(3)
    with col1:
        trestbps = st.number_input("Resting BP (trestbps)", min_value=50, max_value=250, value=130)
    with col2:
        chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=700, value=220)
    with col3:
        thalch = st.number_input("Max Heart Rate (thalch)", min_value=50, max_value=250, value=150)

    col1, col2, col3 = st.columns(3)
    with col1:
        cp = st.selectbox("Chest Pain Type (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    with col2:
        restecg = st.selectbox("Rest ECG (restecg)", ["normal", "lv hypertrophy", "st-t abnormality"])
    with col3:
        slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])

    col1, col2, col3 = st.columns(3)
    with col1:
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [False, True])
    with col2:
        exang = st.selectbox("Exercise Induced Angina (exang)", [False, True])
    with col3:
        ca = st.number_input("Major Vessels (ca)", min_value=0, max_value=4, value=0)

    col1, col2 = st.columns(2)
    with col1:
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col2:
        thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])

    submitted = st.form_submit_button("Predict")


if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    x = preprocessor.transform(input_df)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = x.astype("float32")

    prob = float(model.predict(x, verbose=0)[0][0])
    pred = int(prob >= 0.5)
    label = "Disease" if pred == 1 else "No Disease"

    st.write("")
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prediction", label)
    with c2:
        st.metric("Probability (disease)", f"{prob:.4f}")

    with st.expander("Show input data"):
        st.dataframe(input_df, use_container_width=True)

    log_row = input_df.iloc[0].to_dict()
    log_row.update({
        "probability": prob,
        "prediction": pred,
        "prediction_label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    })
    append_log(log_row)

    st.success(f"Saved input + prediction to `{LOG_FILE}`")

st.write("")
st.subheader("Prediction Logs")

if os.path.exists(LOG_FILE):
    logs = pd.read_csv(LOG_FILE)
    st.dataframe(logs.tail(20), use_container_width=True)
    st.download_button(
        "Download logs CSV",
        data=logs.to_csv(index=False).encode("utf-8"),
        file_name="predictions_log.csv",
        mime="text/csv",
    )
else:
    st.info("No logs yet. Make a prediction to create the log file.")
