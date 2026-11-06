import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import date, datetime

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #07111f;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1728;
        border-right: 1px solid #1e3a5f;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p, label, span {
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: #111f33;
        border: 1px solid #263b56;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    div.stButton > button:focus,
    div.stFormSubmitButton > button:focus {
        outline: none;
        box-shadow: none;
    }

    .hero-card {
        background: linear-gradient(120deg, #102747, #142d4d);
        border: 1px solid #29496e;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
    }

    .result-card {
        background: #142b49;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
    }

    hr {
        border-color: #263b56;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #263b56;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATASET LOADING
# =========================================================

@st.cache_data
def load_csv_file(file_path):
    return pd.read_csv(file_path)


def find_dataset_file(dataset_type):

    search_dirs = [
        RAW_DIR,
        PROCESSED_DIR,
        DATA_DIR
    ]

    all_files = []

    for folder in search_dirs:
        if folder.exists():
            all_files.extend(folder.rglob("*.csv"))

    if dataset_type == "Credit Card Fraud Dataset":

        possible_names = [
            "credit_card_fraud_dataset.csv",
            "credit_card_fraud.csv"
        ]

    else:

        possible_names = [
            "fraud_detection_dataset.csv",
            "fraud_detection.csv"
        ]

    for name in possible_names:

        for file_path in all_files:

            if file_path.name.lower() == name.lower():
                return file_path

    return None


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def get_fraud_column(data):

    possible_columns = [
        "IsFraud",
        "Fraudulent",
        "is_fraud",
        "fraud"
    ]

    for col in possible_columns:
        if col in data.columns:
            return col

    return None


def get_amount_column(data):

    possible_columns = [
        "Amount",
        "Transaction_Amount",
        "total_value"
    ]

    for col in possible_columns:
        if col in data.columns:
            return col

    return None


def get_date_column(data):

    possible_columns = [
        "TransactionDate",
        "Transaction_Date",
        "date"
    ]

    for col in possible_columns:
        if col in data.columns:
            return col

    return None


def get_column(data, possible_names):

    for col in possible_names:
        if col in data.columns:
            return col

    return None


def prepare_fraud_data(data):

    fraud_col = get_fraud_column(data)

    if fraud_col is None:
        return None

    result = data.copy()

    result[fraud_col] = pd.to_numeric(
        result[fraud_col],
        errors="coerce"
    )

    result = result.dropna(subset=[fraud_col])

    return result


def show_kpis(data):

    fraud_col = get_fraud_column(data)
    amount_col = get_amount_column(data)

    total = len(data)

    if fraud_col:

        fraud_values = pd.to_numeric(
            data[fraud_col],
            errors="coerce"
        )

        fraud_count = int(fraud_values.eq(1).sum())
        normal_count = int(fraud_values.eq(0).sum())

        fraud_rate = (
            fraud_count / total * 100
            if total > 0 else 0
        )

    else:

        fraud_count = 0
        normal_count = 0
        fraud_rate = 0

    if amount_col:

        amount_values = pd.to_numeric(
            data[amount_col],
            errors="coerce"
        )

        total_amount = amount_values.sum()
        average_amount = amount_values.mean()

    else:

        total_amount = 0
        average_amount = 0

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{fraud_count:,}")
    c3.metric("Normal Cases", f"{normal_count:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c5.metric("Average Amount", f"{average_amount:,.2f}")

    st.caption(
        f"Total Transaction Amount: {total_amount:,.2f}"
    )


def show_fraud_chart(data):

    fraud_col = get_fraud_column(data)

    if fraud_col is None:
        st.info("Fraud label column not found.")
        return

    values = pd.to_numeric(
        data[fraud_col],
        errors="coerce"
    )

    counts = values.value_counts().sort_index()

    chart_data = pd.DataFrame({
        "Transaction Type": [
            "Normal" if x == 0 else "Fraud"
            for x in counts.index
        ],
        "Count": counts.values
    })

    fig = px.pie(
        chart_data,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        title="Fraud vs Normal Transactions"
    )

    fig.update_layout(
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font_color="#e5e7eb"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_model_info(model_path):

    if not model_path.exists():

        st.warning(f"Model not found: {model_path.name}")
        return

    try:

        saved_data = joblib.load(model_path)

        st.write(
            "**Model:**",
            type(saved_data.get("model")).__name__
        )

        st.write(
            "**Features:**",
            len(saved_data.get("features", []))
        )

        threshold = saved_data.get("threshold")

        if threshold is not None:
            st.write(
                "**Saved Threshold:**",
                f"{threshold:.4f}"
            )

    except Exception as e:

        st.warning(f"Could not load model information: {e}")


def show_prediction_result(
    prediction,
    probability,
    threshold
):

    result_text = (
        "FRAUDULENT TRANSACTION"
        if prediction == 1
        else "NORMAL TRANSACTION"
    )

    st.markdown(
        f"""
        <div class="result-card">
            <h2 style="color:#ffffff;">
                {result_text}
            </h2>
            <p style="color:#dbeafe;">
                Fraud Probability:
                <b>{probability * 100:.2f}%</b>
            </p>
            <p style="color:#dbeafe;">
                Model Decision Threshold:
                <b>{threshold * 100:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This result is a model-based risk estimate, "
        "not a guarantee that a transaction is safe."
    )


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🛡️ FraudShield AI")
    st.caption("Enterprise Fraud Detection Platform")

    st.divider()

    pages = [
        "Executive Dashboard",
        "Dataset Overview",
        "EDA Dashboard",
        "Feature Engineering",
        "Fraud Intelligence",
        "Model Comparison",
        "Fraud Prediction",
        "Fraud Analytics",
        "Detailed Investigation",
        "Explainable AI",
        "Real-Time Monitoring",
        "Fraud Forecasting",
        "Graph Analytics"
    ]

    page = st.radio(
        "Navigation",
        pages
    )

    st.divider()

    st.subheader("Dashboard Filters")

    dataset_choice = st.selectbox(
        "Select Dataset",
        [
            "Credit Card Fraud Dataset",
            "Fraud Detection Dataset"
        ]
    )

    dataset_path = find_dataset_file(dataset_choice)

    df = None

    if dataset_path:

        try:
            df = load_csv_file(dataset_path)

        except Exception as e:
            st.warning(f"Dataset loading error: {e}")

    else:

        st.warning(
            "Dataset CSV not found. Check data/raw or data/processed."
        )

    if df is not None:

        date_col = get_date_column(df)

        if date_col:

            date_values = pd.to_datetime(
                df[date_col],
                errors="coerce"
            ).dropna()

            if not date_values.empty:

                min_date = date_values.min().date()
                max_date = date_values.max().date()

                selected_dates = st.date_input(
                    "Transaction Date Range",
                    value=(min_date, max_date)
                )

                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

                    start_date, end_date = selected_dates

                    date_series = pd.to_datetime(
                        df[date_col],
                        errors="coerce"
                    ).dt.date

                    df = df[
                        date_series.between(
                            start_date,
                            end_date
                        )
                    ]

        st.caption(
            f"Selected records: {len(df):,}"
        )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center;padding:10px;">
            <h2>🛡️</h2>
            <b>FraudShield AI</b>
            <p class="small-muted">
                AI-Powered Financial Security
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 6. EXECUTIVE DASHBOARD
# =========================================================

if page == "Executive Dashboard":

    st.title("🏠 Executive Dashboard")

    st.markdown(
        """
        <div class="hero-card">
            <h2>Financial Fraud Detection Platform</h2>
            <p>
                Monitor transactions, analyze fraud patterns,
                investigate suspicious activity and predict
                financial fraud using machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df is None or df.empty:

        st.warning("No dataset available. Check your CSV files.")

    else:

        show_kpis(df)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            show_fraud_chart(df)

        with col2:

            amount_col = get_amount_column(df)

            if amount_col:

                amount_data = pd.to_numeric(
                    df[amount_col],
                    errors="coerce"
                ).dropna()

                fig = px.histogram(
                    amount_data,
                    title="Transaction Amount Distribution",
                    nbins=30
                )

                fig.update_layout(
                    paper_bgcolor="#07111f",
                    plot_bgcolor="#07111f",
                    font_color="#e5e7eb"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.divider()

        st.subheader("AI Fraud Detection Pipeline")

        pipeline_cols = st.columns(7)

        pipeline_steps = [
            "📥 Data",
            "🧹 Clean",
            "📊 EDA",
            "⚖️ SMOTE",
            "⚙️ Features",
            "🤖 Models",
            "🚨 Detect"
        ]

        for col, step in zip(pipeline_cols, pipeline_steps):
            col.markdown(f"### {step}")

        st.divider()

        st.subheader("Platform Features")

        feature_cols = st.columns(3)

        features = [
            "🤖 Machine Learning Models",
            "📡 Transaction Monitoring",
            "📈 Fraud Trend Analysis",
            "🎯 Fraud Risk Scoring",
            "🕵️ Fraud Investigation",
            "🕸️ Graph Analytics",
            "📊 Executive Dashboard",
            "🔍 Explainable AI",
            "📁 Dataset Explorer"
        ]

        for i, feature in enumerate(features):

            with feature_cols[i % 3]:

                st.info(feature)


# =========================================================
# 7. DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        show_kpis(df)

        st.divider()

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("First 10 Records")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

        with col2:

            st.subheader("Statistical Summary")

            st.dataframe(
                df.describe(include="all").T,
                use_container_width=True
            )

        st.subheader("Missing Values")

        missing = df.isnull().sum()

        missing = missing[missing > 0]

        if missing.empty:

            st.success("No missing values found.")

        else:

            st.bar_chart(missing)


# =========================================================
# 8. EDA DASHBOARD
# =========================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        show_kpis(df)

        st.divider()

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Numerical Analysis")

            if numeric_cols:

                selected_numeric = st.selectbox(
                    "Select Numerical Feature",
                    numeric_cols
                )

                fig = px.histogram(
                    df,
                    x=selected_numeric,
                    nbins=30,
                    title=f"Distribution of {selected_numeric}"
                )

                fig.update_layout(
                    paper_bgcolor="#07111f",
                    plot_bgcolor="#07111f",
                    font_color="#e5e7eb"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:

            st.subheader("Categorical Analysis")

            if categorical_cols:

                selected_category = st.selectbox(
                    "Select Categorical Feature",
                    categorical_cols
                )

                category_counts = (
                    df[selected_category]
                    .astype(str)
                    .value_counts()
                    .head(15)
                )

                st.bar_chart(category_counts)

        st.divider()

        st.subheader("Correlation Heatmap")

        if len(numeric_cols) >= 2:

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Numerical Feature Correlation"
            )

            fig.update_layout(
                paper_bgcolor="#07111f",
                font_color="#e5e7eb"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("Not enough numerical columns for correlation.")


# =========================================================
# 9. FEATURE ENGINEERING
# =========================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        st.subheader("Available Features")

        st.write(
            f"Total Features: **{df.shape[1]}**"
        )

        st.dataframe(
            pd.DataFrame({
                "Feature": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values
            }),
            use_container_width=True,
            hide_index=True
        )

        date_col = get_date_column(df)

        if date_col:

            st.subheader("Date-Based Features")

            date_data = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

            feature_preview = pd.DataFrame({
                "Transaction Date": date_data,
                "Month": date_data.dt.month,
                "Day": date_data.dt.day,
                "Weekday": date_data.dt.weekday,
                "Is Weekend": (
                    date_data.dt.weekday >= 5
                ).astype("Int64")
            })

            st.dataframe(
                feature_preview.head(10),
                use_container_width=True
            )

        st.info(
            "This page previews feature engineering. "
            "It does not modify your original training datasets."
        )


# =========================================================
# 10. FRAUD INTELLIGENCE
# =========================================================

elif page == "Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        fraud_col = get_fraud_column(df)
        amount_col = get_amount_column(df)

        if fraud_col:

            fraud_data = df.copy()

            fraud_data[fraud_col] = pd.to_numeric(
                fraud_data[fraud_col],
                errors="coerce"
            )

            fraud_only = fraud_data[
                fraud_data[fraud_col] == 1
            ]

            st.metric(
                "Identified Fraud Cases",
                f"{len(fraud_only):,}"
            )

            st.subheader("Fraud Transaction Records")

            st.dataframe(
                fraud_only.head(100),
                use_container_width=True,
                hide_index=True
            )

            if amount_col and not fraud_only.empty:

                fraud_amount = pd.to_numeric(
                    fraud_only[amount_col],
                    errors="coerce"
                )

                st.metric(
                    "Total Fraud Amount",
                    f"{fraud_amount.sum():,.2f}"
                )

                st.metric(
                    "Average Fraud Amount",
                    f"{fraud_amount.mean():,.2f}"
                )

        else:

            st.info("Fraud label column not found.")


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("⚖️ Model Comparison")

    model_files = {
        "Credit Card Fraud Model":
            MODEL_DIR / "credit_card_fraud_model.pkl",

        "Credit Card SMOTE Model":
            MODEL_DIR / "credit_card_fraud_smote_model.pkl",

        "Fraud Detection Model":
            MODEL_DIR / "fraud_detection_model.pkl",

        "Fraud Detection SMOTE Model":
            MODEL_DIR / "fraud_detection_smote_model.pkl"
    }

    for model_name, model_path in model_files.items():

        with st.expander(model_name):

            show_model_info(model_path)

    st.info(
        "This section displays saved model information. "
        "No new evaluation metrics are calculated here."
    )


# =========================================================
# 12. FRAUD PREDICTION
# =========================================================

elif page == "Fraud Prediction":

    st.title("🚨 Fraud Prediction")

    st.caption(
        "Analyze a transaction using your trained Random Forest model."
    )

    prediction_dataset = st.selectbox(
        "Choose Prediction Dataset",
        [
            "Credit Card Fraud Dataset",
            "Fraud Detection Dataset"
        ],
        key="prediction_dataset"
    )

    if prediction_dataset == "Credit Card Fraud Dataset":

        model_path = MODEL_DIR / "credit_card_fraud_model.pkl"

        st.subheader("Credit Card Transaction Details")

        with st.form("credit_card_prediction_form"):

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

            submitted = st.form_submit_button(
                "🔍 Predict Fraud",
                use_container_width=True
            )

        if submitted:

            input_data = {
                "Amount": amount,
                "MerchantID": merchant_id,
                "TransactionType": transaction_type,
                "Location": location,
                "Transaction_Month": transaction_date.month,
                "Transaction_Day": transaction_date.day,
                "Transaction_Weekday": transaction_date.weekday(),
                "Is_Weekend": int(
                    transaction_date.weekday() >= 5
                )
            }

            try:

                if not model_path.exists():

                    st.warning("Credit card model file not found.")

                else:

                    saved_data = joblib.load(model_path)

                    model = saved_data["model"]
                    threshold = saved_data["threshold"]
                    features = saved_data["features"]

                    input_df = pd.DataFrame(
                        [input_data]
                    ).reindex(columns=features)

                    probability = model.predict_proba(
                        input_df
                    )[0][1]

                    prediction = int(
                        probability >= threshold
                    )

                    show_prediction_result(
                        prediction,
                        probability,
                        threshold
                    )

            except Exception as e:

                st.warning(f"Prediction Error: {e}")

    else:

        model_path = MODEL_DIR / "fraud_detection_model.pkl"

        st.subheader("Fraud Detection Transaction Details")

        with st.form("fraud_detection_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                transaction_id = st.number_input(
                    "Transaction ID",
                    min_value=1,
                    value=1001,
                    key="fd_transaction_id"
                )

                customer_id = st.number_input(
                    "Customer ID",
                    min_value=0,
                    value=1001
                )

                amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=500.0,
                    key="fd_amount"
                )

                merchant_category = st.selectbox(
                    "Merchant Category",
                    [
                        "Retail",
                        "Food",
                        "Travel",
                        "Online",
                        "Other"
                    ]
                )

                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Credit Card",
                        "Debit Card",
                        "UPI",
                        "Cash",
                        "Other"
                    ]
                )

                device_type = st.selectbox(
                    "Device Type",
                    [
                        "Mobile",
                        "Desktop",
                        "Tablet",
                        "Other"
                    ]
                )

            with col2:

                transaction_date = st.date_input(
                    "Transaction Date",
                    value=date.today(),
                    key="fd_date"
                )

                location = st.text_input(
                    "Location",
                    value="India",
                    key="fd_location"
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

            submitted = st.form_submit_button(
                "🔍 Predict Fraud",
                use_container_width=True
            )

        if submitted:

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
                "Is_Weekend": int(
                    transaction_date.weekday() >= 5
                )
            }

            try:

                if not model_path.exists():

                    st.warning("Fraud detection model file not found.")

                else:

                    saved_data = joblib.load(model_path)

                    model = saved_data["model"]
                    threshold = saved_data["threshold"]
                    features = saved_data["features"]

                    input_df = pd.DataFrame(
                        [input_data]
                    ).reindex(columns=features)

                    probability = model.predict_proba(
                        input_df
                    )[0][1]

                    prediction = int(
                        probability >= threshold
                    )

                    show_prediction_result(
                        prediction,
                        probability,
                        threshold
                    )

            except Exception as e:

                st.warning(f"Prediction Error: {e}")


# =========================================================
# 13. FRAUD ANALYTICS
# =========================================================

elif page == "Fraud Analytics":

    st.title("📊 Fraud Analytics")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        fraud_col = get_fraud_column(df)
        amount_col = get_amount_column(df)

        if fraud_col:

            fraud_values = pd.to_numeric(
                df[fraud_col],
                errors="coerce"
            )

            fraud_count = int(
                fraud_values.eq(1).sum()
            )

            normal_count = int(
                fraud_values.eq(0).sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                f"{len(df):,}"
            )

            col2.metric(
                "Fraud Transactions",
                f"{fraud_count:,}"
            )

            col3.metric(
                "Normal Transactions",
                f"{normal_count:,}"
            )

            st.divider()

            show_fraud_chart(df)

            if amount_col:

                st.subheader("Average Transaction Amount by Class")

                chart_data = df.copy()

                chart_data[amount_col] = pd.to_numeric(
                    chart_data[amount_col],
                    errors="coerce"
                )

                chart_data[fraud_col] = pd.to_numeric(
                    chart_data[fraud_col],
                    errors="coerce"
                )

                amount_summary = (
                    chart_data.groupby(fraud_col)[amount_col]
                    .mean()
                    .rename(index={
                        0: "Normal",
                        1: "Fraud"
                    })
                )

                st.bar_chart(amount_summary)

        else:

            st.info("Fraud label column not found.")


# =========================================================
# 14. DETAILED INVESTIGATION
# =========================================================

elif page == "Detailed Investigation":

    st.title("🕵️ Detailed Investigation")

    st.caption(
        "Investigate suspicious transactions using advanced filters."
    )

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        customer_col = get_column(
            df,
            ["Customer_ID", "CustomerID", "customer_id"]
        )

        transaction_col = get_column(
            df,
            [
                "Transaction_ID",
                "TransactionID",
                "receipt_id"
            ]
        )

        fraud_col = get_fraud_column(df)
        amount_col = get_amount_column(df)

        col1, col2 = st.columns(2)

        with col1:

            customer_search = st.text_input(
                "Search Customer ID"
            )

        with col2:

            transaction_search = st.text_input(
                "Search Transaction ID"
            )

        col1, col2 = st.columns(2)

        with col1:

            only_fraud = st.checkbox(
                "Only Fraud Transactions"
            )

        with col2:

            only_high_amount = st.checkbox(
                "Only High Amount (>5000)"
            )

        filtered_df = df.copy()

        if customer_search and customer_col:

            filtered_df = filtered_df[
                filtered_df[customer_col]
                .astype(str)
                .str.contains(
                    customer_search,
                    case=False,
                    na=False
                )
            ]

        if transaction_search and transaction_col:

            filtered_df = filtered_df[
                filtered_df[transaction_col]
                .astype(str)
                .str.contains(
                    transaction_search,
                    case=False,
                    na=False
                )
            ]

        if only_fraud and fraud_col:

            filtered_df = filtered_df[
                pd.to_numeric(
                    filtered_df[fraud_col],
                    errors="coerce"
                ) == 1
            ]

        if only_high_amount and amount_col:

            filtered_df = filtered_df[
                pd.to_numeric(
                    filtered_df[amount_col],
                    errors="coerce"
                ) > 5000
            ]

        st.divider()

        fraud_count = 0

        if fraud_col:

            fraud_count = int(
                pd.to_numeric(
                    filtered_df[fraud_col],
                    errors="coerce"
                ).eq(1).sum()
            )

        total_amount = 0
        average_amount = 0

        if amount_col:

            amounts = pd.to_numeric(
                filtered_df[amount_col],
                errors="coerce"
            )

            total_amount = amounts.sum()
            average_amount = amounts.mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Transactions",
            f"{len(filtered_df):,}"
        )

        c2.metric(
            "Fraud Cases",
            f"{fraud_count:,}"
        )

        c3.metric(
            "Total Amount",
            f"{total_amount:,.2f}"
        )

        c4.metric(
            "Average Amount",
            f"{average_amount:,.2f}"
        )

        st.divider()

        st.subheader("Investigation Results")

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Investigation Results",
            data=csv_data,
            file_name="fraud_investigation_results.csv",
            mime="text/csv"
        )


# =========================================================
# 15. EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")

    model_choice = st.selectbox(
        "Select Model",
        [
            "Credit Card Fraud Model",
            "Fraud Detection Model"
        ]
    )

    if model_choice == "Credit Card Fraud Model":

        model_path = MODEL_DIR / "credit_card_fraud_model.pkl"

    else:

        model_path = MODEL_DIR / "fraud_detection_model.pkl"

    if model_path.exists():

        try:

            saved_data = joblib.load(model_path)

            model = saved_data["model"]
            features = saved_data.get("features", [])

            estimator = model

            if hasattr(model, "named_steps"):

                estimator = list(
                    model.named_steps.values()
                )[-1]

            if hasattr(estimator, "feature_importances_"):

                importances = estimator.feature_importances_

                if len(importances) == len(features):

                    importance_df = pd.DataFrame({
                        "Feature": features,
                        "Importance": importances
                    }).sort_values(
                        "Importance",
                        ascending=False
                    )

                    st.subheader("Feature Importance")

                    fig = px.bar(
                        importance_df.head(15),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 15 Important Features"
                    )

                    fig.update_layout(
                        paper_bgcolor="#07111f",
                        plot_bgcolor="#07111f",
                        font_color="#e5e7eb"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "Feature count does not match model importance values."
                    )

            else:

                st.info(
                    "Feature importance is not directly available "
                    "from this saved model."
                )

        except Exception as e:

            st.warning(f"Model inspection error: {e}")

    else:

        st.warning("Selected model file not found.")


# =========================================================
# 16. REAL-TIME MONITORING
# =========================================================

elif page == "Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        st.subheader("Latest Transaction Records")

        date_col = get_date_column(df)

        monitoring_df = df.copy()

        if date_col:

            monitoring_df[date_col] = pd.to_datetime(
                monitoring_df[date_col],
                errors="coerce"
            )

            monitoring_df = monitoring_df.sort_values(
                date_col,
                ascending=False
            )

        st.dataframe(
            monitoring_df.head(50),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This page displays records from your selected CSV dataset. "
            "It is not connected to a live banking transaction stream."
        )


# =========================================================
# 17. FRAUD FORECASTING
# =========================================================

elif page == "Fraud Forecasting":

    st.title("📈 Fraud Forecasting & Trend Analysis")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        date_col = get_date_column(df)
        fraud_col = get_fraud_column(df)

        if date_col and fraud_col:

            trend_df = df[
                [date_col, fraud_col]
            ].copy()

            trend_df[date_col] = pd.to_datetime(
                trend_df[date_col],
                errors="coerce"
            )

            trend_df[fraud_col] = pd.to_numeric(
                trend_df[fraud_col],
                errors="coerce"
            )

            trend_df = trend_df.dropna()

            if not trend_df.empty:

                trend_df["Date"] = (
                    trend_df[date_col].dt.date
                )

                daily_fraud = trend_df.groupby(
                    "Date"
                )[fraud_col].sum()

                daily_transactions = trend_df.groupby(
                    "Date"
                ).size()

                st.subheader("Daily Fraud Transactions")

                st.line_chart(daily_fraud)

                st.subheader("Daily Transaction Volume")

                st.line_chart(daily_transactions)

            else:

                st.info("No valid date records found.")

        else:

            st.info(
                "Date or fraud label column is missing."
            )

        st.info(
            "This page displays historical fraud trends. "
            "It does not generate future predictions."
        )


# =========================================================
# 18. GRAPH ANALYTICS
# =========================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")

    if df is None or df.empty:

        st.warning("Dataset not available.")

    else:

        possible_entities = [
            "Customer_ID",
            "CustomerID",
            "MerchantID",
            "Merchant_ID",
            "Merchant_Category",
            "Location",
            "Device_Type",
            "Payment_Method"
        ]

        available_entities = [
            col for col in possible_entities
            if col in df.columns
        ]

        if available_entities:

            selected_entity = st.selectbox(
                "Select Entity",
                available_entities
            )

            entity_counts = (
                df[selected_entity]
                .astype(str)
                .value_counts()
                .head(15)
            )

            st.subheader(
                f"Top {selected_entity} Values"
            )

            st.bar_chart(entity_counts)

            st.dataframe(
                entity_counts.rename("Transaction Count"),
                use_container_width=True
            )

        else:

            st.info(
                "No supported entity columns found in this dataset."
            )

        st.info(
            "This page shows entity frequency analysis. "
            "It does not calculate actual network connections."
        )


# =========================================================
# 19. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:18px;
        color:#94a3b8;
    ">
        <h3>🛡️ FraudShield AI</h3>
        <p>
            AI-Powered Financial Fraud Detection & Transaction Monitoring System
        </p>
        <p>
            Built with Python | Pandas | Scikit-learn | Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
