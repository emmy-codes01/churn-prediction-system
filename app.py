import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="ML Project - Customer Churn Predictor",
    page_icon="",
    layout="centered"
)

st.title("Customer Churn Predictor")
st.subheader("ML Project")
st.caption("Predict whether a customer is likely to leave the telecom service.")

st.divider()

# ---------------------------
# Customer Details
# ---------------------------

customer_name = st.text_input(
    "Customer Name",
    placeholder="e.g. Elias"
)

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    
st.divider()

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Customer", use_container_width=True):

    input_dict = {col: 0 for col in model_columns}

    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly_charges

    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1

    if f"InternetService_{internet}" in input_dict:
        input_dict[f"InternetService_{internet}"] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f":material/warning: **{customer_name or 'Customer'}** is likely to leave."
        )

    else:

        st.success(
            f":material/check_circle: **{customer_name or 'Customer'}** is likely to stay."
        )

    st.progress(float(probability))

    st.metric(
        " Customer's Exit Probability",
        f"{probability:.1%}"
    )

    st.info(
        f"""
**Customer:** {customer_name or "Not Provided"}

**Tenure:** {tenure} months

**Monthly Charges:** ${monthly_charges:.2f}

**Contract:** {contract}

**Internet Service:** {internet}
"""
    )
