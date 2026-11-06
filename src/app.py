
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIRS = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data"
]

MODEL_DIR = BASE_DIR / "models"

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: #07111f;
    color: #e8eef7;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030914 0%, #0b1b2e 100%);
    border-right: 1px solid #1e3550;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #dbeafe;
}

/* SIDEBAR RADIO */

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 9px;
    padding: 8px 10px;
    transition: 0.2s;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #142b43;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px;
    font-weight: 500;
}

/* HEADINGS */

h1, h2, h3 {
    color: #f8fafc !important;
}

/* HERO BANNER */

.hero {
    background: linear-gradient(115deg, #0799e8 0%, #1761e8 55%, #174bd0 100%);
    padding: 42px 42px;
    border-radius: 25px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0, 80, 200, 0.22);
}

.hero-title {
    font-size: 49px;
    font-weight: 850;
    color: white;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-subtitle {
    font-size: 25px;
    font-weight: 650;
    color: #ffffff;
    margin-top: 22px;
}

.hero-description {
    font-size: 16px;
    color: #dbeafe;
    margin-top: 23px;
    line-height: 1.7;
}

/* KPI */

div[data-testid="stMetric"] {
    background: #101d2e;
    border: 1px solid #203650;
    padding: 19px 20px;
    border-radius: 13px;
    min-height: 112px;
}

div[data-testid="stMetricLabel"] {
    color: #a8b8cc !important;
    font-size: 14px;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 30px;
    font-weight: 800;
}

/* CARDS */

.custom-card {
    background: linear-gradient(145deg, #111e30, #0d1929);
    border: 1px solid #203650;
    border-radius: 16px;
    padding: 23px;
    margin-top: 14px;
    margin-bottom: 15px;
}

.card-heading {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 750;
    margin-bottom: 15px;
}

.card-text {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.8;
}

/* PIPELINE */

.pipeline-container {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    gap: 9px;
    width: 100%;
    overflow-x: auto;
    padding: 15px 2px;
}

.pipeline-step {
    flex: 1 0 auto;
    min-width: 125px;
    white-space: nowrap;
    text-align: center;
    background: #142c45;
    border: 1px solid #2e5278;
    border-radius: 10px;
    padding: 15px 10px;
    color: #eaf4ff;
    font-size: 13px;
    font-weight: 650;
}

.pipeline-arrow {
    flex: 0 0 auto;
    color: #38bdf8;
    font-size: 20px;
}

/* FEATURES */

.feature-container {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    gap: 12px;
    overflow-x: auto;
    width: 100%;
    padding: 12px 2px;
}

.feature-item {
    flex: 1 0 auto;
    min-width: 130px;
    white-space: nowrap;
    text-align: center;
    padding: 15px 10px;
    border-radius: 10px;
    background: #142c45;
    border: 1px solid #2e5278;
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 600;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 8px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #0369a1;
    color: white;
}

/* INPUTS */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #101d2e;
    color: white;
}

/* TABLES */

[data-testid="stDataFrame"] {
    border: 1px solid #203650;
    border-radius: 10px;
}

/* DIVIDER */

hr {
    border-color: #203650;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8295ac;
    padding: 25px 5px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATASET AND MODEL FUNCTIONS
# =========================================================

def find_dataset(filename):
    for folder in DATA_DIRS:
        path = folder / filename
        if path.exists():
            return path

    for folder in DATA_DIRS:
        if folder.exists():
            matches = list(folder.rglob(filename))
            if matches:
                return matches[0]

    return None


@st.cache_data(show_spinner=False)
def load_dataset(filename):
    path = find_dataset(filename)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def get_target(df):
    if df is None:
        return None

    candidates = [
        "IsFraud",
        "Fraudulent",
        "is_fraud",
        "fraud",
        "Class"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_amount_column(df):
    candidates = [
        "Amount",
        "Transaction_Amount",
        "total_value"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_date_column(df):
    candidates = [
        "TransactionDate",
        "Transaction_Date",
        "date"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def show_table(df, rows=100):
    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


def model_artifact_count():
    if not MODEL_DIR.exists():
        return 0

    return len(list(MODEL_DIR.glob("*.pkl")))


def notebook_count():
    return len(list(BASE_DIR.rglob("*.ipynb")))


def get_model_name(filename):
    saved = load_model(filename)

    if saved is None:
        return None

    model = saved.get("model") if isinstance(saved, dict) else saved

    if model is None:
        return None

    return type(model).__name__


def get_best_model_label():
    model_names = [
        get_model_name("credit_card_fraud_model.pkl"),
        get_model_name("fraud_detection_model.pkl")
    ]

    model_names = [x for x in model_names if x]

    if not model_names:
        return "Not Available"

    return model_names[0].replace("Classifier", "")


# =========================================================
# 4. LOAD PROJECT DATA
# =========================================================

credit_df = load_dataset("credit_card_fraud_dataset.csv")
fraud_df = load_dataset("fraud_detection_dataset.csv")

dataset_options = {}

if credit_df is not None:
    dataset_options["Credit Card Fraud Dataset"] = credit_df

if fraud_df is not None:
    dataset_options["Fraud Detection Dataset"] = fraud_df


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div style="text-align:center; padding:12px 0 20px 0;">
    <div style="font-size:43px;">🛡️</div>
    <div style="font-size:22px; font-weight:800; color:white;">
        FraudShield AI
    </div>
    <div style="font-size:11px; color:#8ca5c1;">
        FRAUD INTELLIGENCE PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

pages = {
    "🏠 Executive Dashboard": "Executive Dashboard",
    "📁 Dataset Overview": "Dataset Overview",
    "📊 EDA Dashboard": "EDA Dashboard",
    "⚙️ Feature Engineering": "Feature Engineering",
    "🕵️ Fraud Intelligence": "Fraud Intelligence",
    "🤖 Model Comparison": "Model Comparison",
    "🚨 Fraud Prediction": "Fraud Prediction",
    "📈 Fraud Analytics": "Fraud Analytics",
    "🔎 Detailed Investigation": "Detailed Investigation",
    "🧠 Explainable AI": "Explainable AI",
    "📡 Real-Time Monitoring": "Real-Time Monitoring",
    "🔮 Fraud Forecasting": "Fraud Forecasting",
    "🔗 Graph Analytics": "Graph Analytics"
}

selected_page = st.sidebar.radio(
    "NAVIGATION",
    list(pages.keys()),
    index=0
)

page = pages[selected_page]

st.sidebar.divider()

st.sidebar.markdown("""
<div style="text-align:center; padding:15px 5px;">
    <div style="font-size:32px;">🛡️</div>
    <b style="color:white;">FraudShield AI</b>
    <p style="font-size:12px; color:#8ca5c1;">
        Enterprise Fraud Detection Platform
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. EXECUTIVE DASHBOARD - HOMEPAGE
# =========================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🛡️ FraudShield AI</div>

        <div class="hero-subtitle">
            Enterprise Fraud Detection & Risk Intelligence Platform
        </div>

        <div class="hero-description">
            Real-Time Monitoring • Machine Learning • Risk Scoring •
            Fraud Investigation • Graph Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Actual dataset statistics
    total_transactions = sum(
        len(df) for df in dataset_options.values()
    )

    total_fraud = 0

    for df in dataset_options.values():
        target = get_target(df)

        if target:
            total_fraud += int(
                pd.to_numeric(df[target], errors="coerce").fillna(0).sum()
            )

    total_models = model_artifact_count()
    total_notebooks = notebook_count()
    best_model = get_best_model_label()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    k2.metric(
        "Fraud Records",
        f"{total_fraud:,}"
    )

    k3.metric(
        "ML Models",
        total_models
    )

    k4.metric(
        "Best Model",
        best_model
    )

    st.markdown("""
    <div class="custom-card">
        <div class="card-heading">🚀 Project Overview</div>
        <div class="card-text">
            FraudShield AI is an intelligent financial fraud detection
            platform designed to analyze transaction behavior and identify
            potentially fraudulent activities using Machine Learning.
            <br><br>
            The platform uses transaction amount, customer information,
            merchant category, payment method, device type, location and
            other available dataset features to support fraud analysis.
            <br><br>
            It provides transaction investigation, model-based prediction,
            exploratory data analysis, fraud analytics and feature
            importance insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("⚡ AI Fraud Detection Pipeline")

    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">📥 Data Input</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🧹 Data Cleaning</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">⚙️ Feature Engineering</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🤖 ML Model</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🔍 Fraud Analysis</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">📊 Final Result</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("✨ Platform Features")

    st.markdown("""
    <div class="feature-container">
        <div class="feature-item">📊 Dashboard</div>
        <div class="feature-item">📁 Dataset Analysis</div>
        <div class="feature-item">🤖 ML Prediction</div>
        <div class="feature-item">🔎 Investigation</div>
        <div class="feature-item">📈 Analytics</div>
        <div class="feature-item">🧠 Explainable AI</div>
        <div class="feature-item">📡 Monitoring</div>
        <div class="feature-item">🔗 Graph Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Dataset Summary")

    if dataset_options:

        summary = []

        for name, df in dataset_options.items():

            target = get_target(df)

            fraud_count = (
                int(pd.to_numeric(df[target], errors="coerce").fillna(0).sum())
                if target else 0
            )

            summary.append({
                "Dataset": name,
                "Transactions": len(df),
                "Features": df.shape[1],
                "Fraud Records": fraud_count,
                "Missing Values": int(df.isnull().sum().sum())
            })

        show_table(pd.DataFrame(summary))

    else:
        st.warning(
            "Datasets not found. Please check the data/raw or data/processed folder."
        )

    # Actual fraud distribution across available datasets
    if dataset_options:

        chart_data = []

        for name, df in dataset_options.items():

            target = get_target(df)

            if target:
                fraud_count = int(
                    pd.to_numeric(df[target], errors="coerce").fillna(0).sum()
                )

                chart_data.append({
                    "Dataset": name,
                    "Fraud": fraud_count,
                    "Normal": len(df) - fraud_count
                })

        if chart_data:

            chart_df = pd.DataFrame(chart_data)

            fig = px.bar(
                chart_df,
                x="Dataset",
                y=["Fraud", "Normal"],
                barmode="group",
                title="Fraud vs Normal Records",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 7. DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")
    st.caption("Explore the available transaction datasets.")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        target = get_target(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        st.subheader("Dataset Preview")
        show_table(df, 100)

        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        show_table(info, len(info))

        if target:

            counts = (
                df[target]
                .value_counts()
                .rename(index={0: "Normal", 1: "Fraud"})
                .reset_index()
            )

            counts.columns = ["Transaction Type", "Count"]

            fig = px.pie(
                counts,
                names="Transaction Type",
                values="Count",
                hole=0.5,
                title="Target Distribution",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No datasets found.")


# =========================================================
# 8. EDA DASHBOARD
# =========================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        numeric = df.select_dtypes(include=np.number)

        st.subheader("Numerical Summary")

        if not numeric.empty:
            show_table(numeric.describe().T.reset_index(), 100)

            st.subheader("Correlation Heatmap")

            corr = numeric.corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No numerical columns available.")

        st.subheader("Missing Value Analysis")

        missing = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing.columns = ["Column", "Missing Values"]

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        amount_col = get_amount_column(df)

        if amount_col:

            st.subheader("Transaction Amount Distribution")

            fig = px.histogram(
                df,
                x=amount_col,
                nbins=40,
                title="Transaction Amount Distribution",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Datasets not found.")


# =========================================================
# 9. FEATURE ENGINEERING
# =========================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Features", df.shape[1])
        c2.metric("Numerical Features", len(numeric_cols))
        c3.metric("Categorical Features", len(categorical_cols))

        st.subheader("Numerical Features")
        st.write(numeric_cols)

        st.subheader("Categorical Features")
        st.write(categorical_cols)

        st.subheader("Feature Information")

        feature_info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values,
            "Missing Values": df.isnull().sum().values
        })

        show_table(feature_info, len(feature_info))

        st.info(
            "This page displays the existing feature structure. "
            "It does not modify your original training dataset."
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 10. FRAUD INTELLIGENCE
# =========================================================

elif page == "Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        target = get_target(df)

        if target:

            fraud_df = df[
                pd.to_numeric(df[target], errors="coerce").fillna(0) == 1
            ]

            normal_count = len(df) - len(fraud_df)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Records", f"{len(fraud_df):,}")
            c2.metric("Normal Records", f"{normal_count:,}")

            c3.metric(
                "Fraud Rate",
                f"{len(fraud_df) / len(df) * 100:.2f}%"
            )

            st.subheader("Fraud Transaction Records")
            show_table(fraud_df, 200)

            amount_col = get_amount_column(df)

            if amount_col:

                st.subheader("Fraud Transaction Amount")

                fig = px.histogram(
                    fraud_df,
                    x=amount_col,
                    nbins=30,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison")

    model_files = [
        "credit_card_fraud_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]

    model_rows = []

    for filename in model_files:

        saved = load_model(filename)

        if saved is None:
            continue

        model = saved.get("model") if isinstance(saved, dict) else saved

        threshold = (
            saved.get("threshold")
            if isinstance(saved, dict)
            else None
        )

        features = (
            saved.get("features", [])
            if isinstance(saved, dict)
            else []
        )

        model_rows.append({
            "Model File": filename,
            "Algorithm": type(model).__name__,
            "Threshold": threshold,
            "Features": len(features)
        })

    if model_rows:

        st.metric("Available Model Artifacts", len(model_rows))

        show_table(pd.DataFrame(model_rows), len(model_rows))

    else:
        st.warning("No model files found in models folder.")

    st.info(
        "Accuracy, precision, recall and F1 are not displayed here because "
        "evaluation metric files have not been connected to this page."
    )


# =========================================================
# 12. FRAUD PREDICTION
# =========================================================

elif page == "Fraud Prediction":

    st.title("🚨 Fraud Prediction")
    st.caption("Analyze a transaction using your saved machine learning models.")

    model_options = {
        "Credit Card Fraud Model": "credit_card_fraud_model.pkl",
        "Credit Card Fraud SMOTE Model": "credit_card_fraud_smote_model.pkl",
        "Fraud Detection Model": "fraud_detection_model.pkl",
        "Fraud Detection SMOTE Model": "fraud_detection_smote_model.pkl"
    }

    selected_model = st.selectbox(
        "Choose Prediction Model",
        list(model_options.keys())
    )

    saved = load_model(model_options[selected_model])

    if saved is None:

        st.error("Selected model file not found.")

    elif not isinstance(saved, dict):

        st.error(
            "This model file does not contain the expected model/features dictionary."
        )

    else:

        model = saved.get("model")
        threshold = saved.get("threshold", 0.5)
        features = saved.get("features", [])

        if model is None or not features:

            st.error("Model or feature list is missing.")

        else:

            st.write(f"**Model:** {type(model).__name__}")
            st.write(f"**Threshold:** {threshold}")

            st.subheader("Transaction Details")

            with st.form("prediction_form"):

                values = {}

                cols = st.columns(2)

                for index, feature in enumerate(features):

                    with cols[index % 2]:

                        feature_lower = feature.lower()

                        if any(
                            word in feature_lower
                            for word in [
                                "date",
                                "time",
                                "customer_id",
                                "transaction_id",
                                "merchant",
                                "location",
                                "device",
                                "payment",
                                "category",
                                "keyword",
                                "type"
                            ]
                        ):

                            values[feature] = st.text_input(
                                feature.replace("_", " ").title()
                            )

                        elif "international" in feature_lower:

                            values[feature] = st.selectbox(
                                feature.replace("_", " ").title(),
                                [0, 1]
                            )

                        else:

                            values[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                value=0.0,
                                format="%.4f"
                            )

                submitted = st.form_submit_button(
                    "🔍 Analyze Transaction",
                    use_container_width=True
                )

            if submitted:

                try:

                    input_df = pd.DataFrame([values])
                    input_df = input_df.reindex(columns=features)

                    probability = model.predict_proba(input_df)[0][1]

                    st.divider()
                    st.subheader("Prediction Result")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                    c2.metric(
                        "Decision Threshold",
                        f"{threshold * 100:.2f}%"
                    )

                    if probability >= threshold:
                        st.error("🚨 FRAUD TRANSACTION DETECTED")
                    else:
                        st.success("✅ NORMAL TRANSACTION")

                except Exception as error:

                    st.error(f"Prediction error: {error}")


# =========================================================
# 13. FRAUD ANALYTICS
# =========================================================

elif page == "Fraud Analytics":

    st.title("📈 Fraud Analytics")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        target = get_target(df)

        if target:

            fraud_data = df[
                pd.to_numeric(df[target], errors="coerce").fillna(0) == 1
            ]

            normal_data = df[
                pd.to_numeric(df[target], errors="coerce").fillna(0) == 0
            ]

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Transactions", f"{len(fraud_data):,}")
            c2.metric("Normal Transactions", f"{len(normal_data):,}")

            c3.metric(
                "Fraud Percentage",
                f"{len(fraud_data) / len(df) * 100:.2f}%"
            )

            amount_col = get_amount_column(df)

            if amount_col:

                st.subheader("Fraud vs Normal Amount")

                fig = px.box(
                    df,
                    x=target,
                    y=amount_col,
                    color=target,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

            if "Location" in df.columns:

                st.subheader("Fraud by Location")

                location_data = (
                    fraud_data["Location"]
                    .value_counts()
                    .head(15)
                    .reset_index()
                )

                location_data.columns = [
                    "Location",
                    "Fraud Count"
                ]

                fig = px.bar(
                    location_data,
                    x="Location",
                    y="Fraud Count",
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 14. DETAILED INVESTIGATION
# =========================================================

elif page == "Detailed Investigation":

    st.title("🔎 Detailed Investigation")

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        target = get_target(df)
        amount_col = get_amount_column(df)

        id_columns = [
            c for c in [
                "TransactionID",
                "Transaction_ID",
                "Customer_ID",
                "receipt_id"
            ]
            if c in df.columns
        ]

        search_col = st.selectbox(
            "Search By",
            ["All Columns"] + id_columns
        )

        search_value = st.text_input(
            "Enter Transaction ID or Customer ID"
        )

        c1, c2 = st.columns(2)

        with c1:
            fraud_only = st.checkbox("Fraud transactions only")

        with c2:
            high_amount = st.checkbox("High-value transactions (> 5000)")

        filtered = df.copy()

        if search_value:

            if search_col == "All Columns":

                mask = filtered.astype(str).apply(
                    lambda col: col.str.contains(
                        search_value,
                        case=False,
                        na=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

            else:

                filtered = filtered[
                    filtered[search_col]
                    .astype(str)
                    .str.contains(search_value, case=False, na=False)
                ]

        if fraud_only and target:

            filtered = filtered[
                pd.to_numeric(filtered[target], errors="coerce").fillna(0) == 1
            ]

        if high_amount and amount_col:

            filtered = filtered[
                pd.to_numeric(filtered[amount_col], errors="coerce") > 5000
            ]

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric("Matching Records", f"{len(filtered):,}")

        if target:
            fraud_count = int(
                pd.to_numeric(filtered[target], errors="coerce").fillna(0).sum()
            )
            c2.metric("Fraud Records", f"{fraud_count:,}")

        if amount_col:
            total_amount = pd.to_numeric(
                filtered[amount_col],
                errors="coerce"
            ).sum()

            c3.metric("Total Amount", f"{total_amount:,.2f}")

        st.subheader("Investigation Results")

        show_table(filtered, 500)

        if len(filtered):

            csv_data = filtered.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Investigation Results",
                data=csv_data,
                file_name="fraud_investigation_results.csv",
                mime="text/csv"
            )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 15. EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")
    st.caption("Understand which features influence the saved model.")

    model_file = st.selectbox(
        "Select Model",
        [
            "credit_card_fraud_model.pkl",
            "fraud_detection_model.pkl",
            "credit_card_fraud_smote_model.pkl",
            "fraud_detection_smote_model.pkl"
        ]
    )

    saved = load_model(model_file)

    if saved is None or not isinstance(saved, dict):

        st.warning("Selected model artifact not found or unsupported.")

    else:

        model = saved.get("model")
        features = saved.get("features", [])

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            )

            fig = px.bar(
                importance.head(20).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Feature Importance",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Importance Details")
            show_table(importance, len(importance))

        else:

            st.info(
                "Feature importance is not directly available for this model."
            )


# =========================================================
# 16. REAL-TIME MONITORING
# =========================================================

elif page == "Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    st.info(
        "This page monitors the available historical transaction dataset. "
        "It is not connected to a live banking transaction stream."
    )

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        target = get_target(df)

        if target:

            fraud_count = int(
                pd.to_numeric(df[target], errors="coerce").fillna(0).sum()
            )

            normal_count = len(df) - fraud_count

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Transactions", f"{len(df):,}")
            c2.metric("Fraud Records", f"{fraud_count:,}")
            c3.metric("Normal Records", f"{normal_count:,}")

            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[normal_count, fraud_count],
                hole=0.55,
                title="Transaction Status",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Dataset Records")
            show_table(df.tail(50), 50)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 17. FRAUD FORECASTING
# =========================================================

elif page == "Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")
    st.caption("Historical fraud trend analysis.")

    st.info(
        "This page displays historical trends. It does not generate "
        "future predictions unless a forecasting model is connected."
    )

    if dataset_options:

        selected = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )

        df = dataset_options[selected]

        date_col = get_date_column(df)
        target = get_target(df)

        if date_col and target:

            trend_df = df[[date_col, target]].copy()

            trend_df[date_col] = pd.to_datetime(
                trend_df[date_col],
                errors="coerce"
            )

            trend_df[target] = pd.to_numeric(
                trend_df[target],
                errors="coerce"
            )

            trend_df = trend_df.dropna(
                subset=[date_col, target]
            )

            trend = (
                trend_df
                .set_index(date_col)
                .resample("D")[target]
                .sum()
                .reset_index()
            )

            trend.columns = ["Date", "Fraud Transactions"]

            fig = px.line(
                trend,
                x="Date",
                y="Fraud Transactions",
                markers=True,
                title="Daily Historical Fraud Transactions",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_table(trend, 100)

        else:
            st.warning("Date column or fraud target not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 18. GRAPH ANALYTICS
# =========================================================

elif page == "Graph Analytics":

    st.title("🔗 Graph Analytics")

    st.info(
        "This page analyzes relationships using available transaction "
        "columns. It is not a dedicated graph neural network."
    )

    if fraud_df is not None:

        relationship_options = [
            col for col in [
                "Customer_ID",
                "Merchant_Category",
                "Payment_Method",
                "Device_Type",
                "Location"
            ]
            if col in fraud_df.columns
        ]

        if relationship_options:

            selected = st.selectbox(
                "Select Relationship Feature",
                relationship_options
            )

            counts = (
                fraud_df[selected]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                selected,
                "Transaction Count"
            ]

            fig = px.bar(
                counts,
                x=selected,
                y="Transaction Count",
                title=f"Transaction Relationships by {selected}",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_table(counts, 20)

        else:
            st.warning("Relationship columns not available.")

    else:
        st.warning("Fraud Detection dataset not found.")


# =========================================================
# 19. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🛡️ <b>FraudShield AI</b>
    <br>
    Enterprise Fraud Detection & Risk Intelligence Platform
    <br>
    Built with Python • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)
