
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import date

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Page Configuration
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# Project Branding
st.title("🛡️ FraudShield AI")
st.subheader(
    "AI-Powered Financial Fraud Detection & Transaction Monitoring System"
)

st.info("Select a dataset to start fraud prediction.")

dataset = st.selectbox(
    "Choose Dataset",
    [
        "Credit Card Fraud Dataset",
        "Fraud Detection Dataset"
    ]
)

# Select model
if dataset == "Credit Card Fraud Dataset":
    model_path = MODEL_DIR / "credit_card_fraud_model.pkl"
else:
    model_path = MODEL_DIR / "fraud_detection_model.pkl"

st.divider()
st.header("🔍 Transaction Details")

input_data = None

# --------------------------------------------------
# DATASET 1: CREDIT CARD FRAUD
# --------------------------------------------------

if dataset == "Credit Card Fraud Dataset":

    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.number_input(
            "Transaction ID",
            min_value=1,
            value=1001
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=500.0
        )

        merchant_id = st.number_input(
            "Merchant ID",
            min_value=0,
            value=101
        )

    with col2:
        transaction_date = st.date_input(
            "Transaction Date",
            value=date.today()
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["Online", "POS", "ATM"]
        )

        location = st.text_input(
            "Location",
            value="India"
        )

    predict_button = st.button(
        "Predict Fraud",
        key="credit_predict"
    )

    if predict_button:

        input_data = {
            "Amount": amount,
            "MerchantID": merchant_id,
            "TransactionType": transaction_type,
            "Location": location,
            "Transaction_Month": transaction_date.month,
            "Transaction_Day": transaction_date.day,
            "Transaction_Weekday": transaction_date.weekday(),
            "Is_Weekend": int(transaction_date.weekday() >= 5)
        }


# --------------------------------------------------
# DATASET 2: FRAUD DETECTION
# --------------------------------------------------

else:

    col1, col2 = st.columns(2)

    with col1:

        transaction_id = st.number_input(
            "Transaction ID",
            min_value=1,
            value=1001
        )

        customer_id = st.text_input(
            "Customer ID",
            value="CUST1001"
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=500.0
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            ["Retail", "Food", "Travel", "Online", "Other"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            ["Credit Card", "Debit Card", "UPI", "Cash"]
        )

        device_type = st.selectbox(
            "Device Type",
            ["Mobile", "Desktop", "Tablet"]
        )

    with col2:

        transaction_date = st.date_input(
            "Transaction Date",
            value=date.today()
        )

        location = st.text_input(
            "Location",
            value="India"
        )

        is_international = st.selectbox(
            "International Transaction",
            [0, 1]
        )

        previous_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=0
        )

        average_spend = st.number_input(
            "Average Spend",
            min_value=0.0,
            value=1000.0
        )

        account_age = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=100
        )

        suspicious_keyword = st.selectbox(
            "Suspicious Keyword",
            [0, 1]
        )

    predict_button = st.button(
        "Predict Fraud",
        key="fraud_predict"
    )

    if predict_button:

        input_data = {
            "Customer_ID": customer_id,
            "Transaction_Amount": amount,
            "Merchant_Category": merchant_category,
            "Payment_Method": payment_method,
            "Device_Type": device_type,
            "Location": location,
            "Is_International": is_international,
            "Previous_Transactions": previous_transactions,
            "Average_Spend": average_spend,
            "Account_Age_Days": account_age,
            "Suspicious_Keyword": suspicious_keyword,
            "Transaction_Month": transaction_date.month,
            "Transaction_Day": transaction_date.day,
            "Transaction_Hour": 0,
            "Transaction_Weekday": transaction_date.weekday(),
            "Is_Weekend": int(transaction_date.weekday() >= 5)
        }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if input_data is not None:

    try:

        if not model_path.exists():
            st.error(
                "Model file not found. Please train the model first."
            )

        else:

            saved_data = joblib.load(model_path)

            model = saved_data["model"]
            threshold = saved_data["threshold"]
            features = saved_data["features"]

            input_df = pd.DataFrame([input_data])

            # Ensure exact original training columns and order
            input_df = input_df.reindex(
                columns=features
            )

            # Pipeline handles missing values and encoding
            fraud_probability = model.predict_proba(
                input_df
            )[0][1]

            prediction = int(
                fraud_probability >= threshold
            )

            st.divider()
            st.header("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("⚠️ FRAUD TRANSACTION DETECTED!")
                else:
                    st.success("✅ NORMAL TRANSACTION")

            with col2:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

            st.metric(
                "Model Decision Threshold",
                f"{threshold * 100:.2f}%"
            )

            st.caption(
                "Prediction is based on the trained model "
                "and its validation-selected threshold. "
                "It is not a guarantee that a transaction is safe."
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")
