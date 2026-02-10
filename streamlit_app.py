import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "telco_churn_model.joblib"
SCALER_PATH = "telco_scaler.joblib"

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📉 Telco Customer Churn Prediction")
st.caption("User-friendly churn prediction app (interactive inputs + trained ML model)")

# -------------------------------
# Load model + scaler
# -------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Missing model/scaler files. Ensure telco_churn_model.joblib and telco_scaler.joblib exist.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# Helper functions
# -------------------------------
def yn_toggle(label, default=False, help_text=None):
    """User-friendly Yes/No toggle, returns 'Yes' or 'No'."""
    val = st.toggle(label, value=default, help=help_text)
    return "Yes" if val else "No"

def parse_totalcharges(x):
    s = str(x).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def build_input_df(inputs: dict) -> pd.DataFrame:
    """Build raw input dataframe with original feature names."""
    df = pd.DataFrame([inputs])
    return df

def preprocess_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match notebook preprocessing:
    - Convert TotalCharges to numeric
    - One-hot encode using get_dummies(drop_first=True)
    - Align columns to training features (model.feature_names_in_)
    - Scale numeric columns using saved scaler
    """
    df = raw_df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # If TotalCharges is blank, auto-calc using MonthlyCharges * tenure (reasonable approximation)
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].clip(lower=1))

    # One-hot encode categorical variables
    X = pd.get_dummies(df, drop_first=True)

    # Align columns
    train_cols = list(model.feature_names_in_)
    for c in train_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[train_cols]

    # Scale numeric columns
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in X.columns]
    if len(num_cols) > 0:
        X.loc[:, num_cols] = scaler.transform(X[num_cols])

    return X

# -------------------------------
# UI Layout
# -------------------------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Decision Threshold",
    0.30, 0.80, 0.50, 0.05,
    help="Lower = catch more churners (higher recall). Higher = fewer false alarms."
)

# Presets (optional but user-friendly)
preset = st.sidebar.selectbox(
    "Quick Preset (optional)",
    ["Custom", "High Churn Risk Example", "Low Churn Risk Example"]
)

# Defaults for custom
defaults = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": np.nan
}

if preset == "High Churn Risk Example":
    defaults.update({
        "tenure": 3,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "MonthlyCharges": 95.0,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    })
elif preset == "Low Churn Risk Example":
    defaults.update({
        "tenure": 60,
        "Contract": "Two year",
        "InternetService": "DSL",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "MonthlyCharges": 55.0,
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)"
    })

st.markdown("### Enter Customer Details")

# Use tabs for user-friendly grouping
tab1, tab2, tab3 = st.tabs(["👤 Profile", "📡 Services", "💳 Billing & Charges"])

with tab1:
    colA, colB, colC = st.columns(3)

    with colA:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)

    with colB:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0 if defaults["SeniorCitizen"] == 0 else 1)

    with colC:
        partner = st.selectbox("Partner", ["No", "Yes"], index=0 if defaults["Partner"] == "No" else 1)

    dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if defaults["Dependents"] == "No" else 1)

with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        phone = st.selectbox("Phone Service", ["No", "Yes"], index=1 if defaults["PhoneService"] == "Yes" else 0)
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"], index=0 if defaults["MultipleLines"] == "No" else 1)

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL","Fiber optic","No"].index(defaults["InternetService"]))
        online_sec = st.selectbox("Online Security", ["No", "Yes"], index=0 if defaults["OnlineSecurity"] == "No" else 1)
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0 if defaults["TechSupport"] == "No" else 1)

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0 if defaults["StreamingTV"] == "No" else 1)
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0 if defaults["StreamingMovies"] == "No" else 1)
        online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=0 if defaults["OnlineBackup"] == "No" else 1)

with tab3:
    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                index=["Month-to-month","One year","Two year"].index(defaults["Contract"]))
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1 if defaults["PaperlessBilling"] == "Yes" else 0)

    with col2:
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ], index=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ].index(defaults["PaymentMethod"]))

    with col3:
        tenure = st.slider("Tenure (months)", 0, 72, int(defaults["tenure"]))
        monthly = st.slider("Monthly Charges", 20.0, 120.0, float(defaults["MonthlyCharges"]), 0.5)

    total_in = st.text_input(
        "Total Charges (Optional — leave blank to auto-calculate)",
        value="" if pd.isna(defaults["TotalCharges"]) else str(defaults["TotalCharges"])
    )

# Convert SeniorCitizen to 0/1 internally, but UI stays Yes/No
senior_val = 1 if senior == "Yes" else 0
total_val = parse_totalcharges(total_in)

# Handle MultipleLines properly if PhoneService=No (dataset has 'No phone service')
if phone == "No":
    multiple_lines_val = "No phone service"
else:
    multiple_lines_val = multiple

# If internet is "No", telco dataset uses "No internet service"
def map_internet_dependent(value, internet_choice):
    if internet_choice == "No":
        return "No internet service"
    return value

online_sec_val = map_internet_dependent(online_sec, internet)
online_backup_val = map_internet_dependent(online_backup, internet)
device_prot_val = map_internet_dependent("No", internet)  # kept but fixed to No for simplicity
tech_support_val = map_internet_dependent(tech_support, internet)
streaming_tv_val = map_internet_dependent(streaming_tv, internet)
streaming_movies_val = map_internet_dependent(streaming_movies, internet)

inputs = {
    "gender": gender,
    "SeniorCitizen": senior_val,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines_val,
    "InternetService": internet,
    "OnlineSecurity": online_sec_val,
    "OnlineBackup": online_backup_val,
    "DeviceProtection": device_prot_val,
    "TechSupport": tech_support_val,
    "StreamingTV": streaming_tv_val,
    "StreamingMovies": streaming_movies_val,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total_val
}

st.markdown("### Preview (what will be fed into the model)")
st.dataframe(pd.DataFrame([inputs]), use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Churn"):
    try:
        raw_df = build_input_df(inputs)
        X_ready = preprocess_for_model(raw_df)

        proba = float(model.predict_proba(X_ready)[:, 1][0])
        pred = "Churn (Yes)" if proba >= threshold else "Churn (No)"

        st.success("Prediction completed.")
        colA, colB, colC = st.columns(3)
        colA.metric("Churn Probability", f"{proba:.3f}")
        colB.metric("Threshold", f"{threshold:.2f}")
        colC.metric("Prediction", pred)

        st.info(
            "Tip: Lower the threshold to catch more churners (higher recall). "
            "Increase it to reduce false alarms."
        )

    except Exception as e:
        st.error("Prediction failed. This usually happens if model columns do not match preprocessing.")
        st.exception(e)
