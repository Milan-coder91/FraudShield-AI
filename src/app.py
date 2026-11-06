
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIRS = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data"
]

MODEL_DIR = BASE_DIR / "models"


# =========================================================
# 2. DATASET FUNCTIONS
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
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_dataset(filename):
    path = find_dataset(filename)
    if path:
        return load_csv(str(path))
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

    for col in [
        "IsFraud",
        "Fraudulent",
        "is_fraud",
        "fraud",
        "Class"
    ]:
        if col in df.columns:
            return col

    return None


def get_amount(df):
    for col in [
        "Amount",
        "Transaction_Amount",
        "total_value"
    ]:
        if col in df.columns:
            return col

    return None


def get_date(df):
    for col in [
        "TransactionDate",
        "Transaction_Date",
        "date"
    ]:
        if col in df.columns:
            return col

    return None


def fraud_count(df):
    target = get_target(df)

    if target is None:
        return 0

    return int(
        pd.to_numeric(df[target], errors="coerce")
        .fillna(0)
        .sum()
    )


def show_table(df, rows=200):
    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


def model_count():
    if MODEL_DIR.exists():
        return len(list(MODEL_DIR.glob("*.pkl")))
    return 0


def get_model_name(filename):
    saved = load_model(filename)

    if saved is None:
        return None

    model = (
        saved.get("model")
        if isinstance(saved, dict)
        else saved
    )

    if model is None:
        return None

    return type(model).__name__.replace("Classifier", "")


def get_best_model():
    for filename in [
        "credit_card_fraud_model.pkl",
        "fraud_detection_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]:
        name = get_model_name(filename)

        if name:
            return name

    return "Not Available"


# =========================================================
# 3. LOAD DATA
# =========================================================

credit_df = load_dataset("credit_card_fraud_dataset.csv")
fraud_df = load_dataset("fraud_detection_dataset.csv")

datasets = {}

if credit_df is not None:
    datasets["Credit Card Fraud Dataset"] = credit_df

if fraud_df is not None:
    datasets["Fraud Detection Dataset"] = fraud_df


# =========================================================
# 4. CSS - BOTH DASHBOARD DESIGNS
# =========================================================

st.markdown("""
<style>

/* GENERAL */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-weight: 750;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #020817 0%,
        #0b1b2e 100%
    );
    border-right: 1px solid #203650;
}

[data-testid="stSidebar"] p {
    color: #dbeafe;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 5px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 10px;
    border-radius: 9px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #18324d;
}

/* HOMEPAGE HERO */

.hero {
    background: linear-gradient(
        115deg,
        #0799e8 0%,
        #1761e8 55%,
        #174bd0 100%
    );
    padding: 48px 42px;
    border-radius: 25px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0, 80, 200, 0.25);
}

.hero-title {
    font-size: 49px;
    font-weight: 850;
    color: white;
    letter-spacing: -1.5px;
}

.hero-subtitle {
    font-size: 25px;
    font-weight: 650;
    color: white;
    margin-top: 22px;
}

.hero-description {
    font-size: 16px;
    color: #dbeafe;
    margin-top: 23px;
    line-height: 1.8;
}

/* HOMEPAGE CARDS */

.project-card {
    background: linear-gradient(
        145deg,
        #111e30,
        #0d1929
    );
    border: 1px solid #203650;
    border-radius: 16px;
    padding: 25px;
    margin-top: 22px;
    margin-bottom: 22px;
}

.project-heading {
    color: white;
    font-size: 28px;
    font-weight: 750;
    margin-bottom: 17px;
}

.project-text {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.9;
}

/* PIPELINE AND FEATURES */

.pipeline-container,
.feature-container {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    gap: 10px;
    width: 100%;
    overflow-x: auto;
    padding: 14px 2px;
}

.pipeline-step,
.feature-item {
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

/* LIGHT EXECUTIVE DASHBOARD */

.light-dashboard {
    background: #ffffff;
    color: #263445;
    padding: 5px;
}

.light-dashboard h1,
.light-dashboard h2,
.light-dashboard h3 {
    color: #263445 !important;
}

/* KPI CARDS */

div[data-testid="stMetric"] {
    background: #f0f5f8;
    border: 1px solid #e1e9ef;
    padding: 18px;
    border-radius: 11px;
    min-height: 105px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #263445 !important;
    font-size: 29px;
    font-weight: 750;
}

/* GENERAL COMPONENTS */

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

hr {
    border-color: #dce5ee;
}

.footer {
    color: #8295ac;
    text-align: center;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div style="text-align:center; padding:12px 0 20px 0;">

    <div style="font-size:43px;">🛡️</div>

    <div style="
        font-size:22px;
        font-weight:800;
        color:white;
    ">
        FraudShield AI
    </div>

    <div style="
        font-size:11px;
        color:#8ca5c1;
    ">
        FRAUD INTELLIGENCE PLATFORM
    </div>

</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

pages = [
    "🏠 app",
    "📊 Executive Dashboard",
    "📁 Dataset Overview",
    "📈 EDA Dashboard",
    "⚙️ Feature Engineering",
    "🕵️ Fraud Intelligence",
    "🤖 Model Comparison",
    "🚨 Fraud Prediction",
    "📊 Fraud Analytics",
    "🔎 Detailed Investigation",
    "🧠 Explainable AI",
    "📡 Real-Time Monitoring",
    "🔮 Fraud Forecasting",
    "🔗 Graph Analytics"
]

page = st.sidebar.radio(
    "NAVIGATION",
    pages,
    index=0
)

st.sidebar.divider()

st.sidebar.markdown("""
<div style="text-align:center; padding:15px 5px;">

    <div style="font-size:34px;">🛡️</div>

    <b style="color:white;">FraudShield AI</b>

    <p style="font-size:12px; color:#8ca5c1;">
        Enterprise Fraud Detection Platform
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. HOMEPAGE - app
# =========================================================

if page == "🏠 app":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🛡️ FraudShield AI
        </div>

        <div class="hero-subtitle">
            Enterprise Fraud Detection & Risk Intelligence Platform
        </div>

        <div class="hero-description">
            Real-Time Monitoring • Machine Learning • Risk Scoring •
            Fraud Investigation • Graph Analytics
        </div>

    </div>
    """, unsafe_allow_html=True)

    total_transactions = sum(
        len(df) for df in datasets.values()
    )

    total_fraud = sum(
        fraud_count(df) for df in datasets.values()
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Transactions", f"{total_transactions:,}")
    k2.metric("Fraud Cases", f"{total_fraud:,}")
    k3.metric("ML Models", model_count())
    k4.metric("Best Model", get_best_model())

    st.markdown("""
    <div class="project-card">

        <div class="project-heading">
            🚀 Project Overview
        </div>

        <div class="project-text">

            FraudShield AI is an intelligent financial fraud detection
            platform designed to identify suspicious and potentially
            fraudulent transactions using Machine Learning.

            <br><br>

            The platform analyzes transaction behavior, customer
            information, merchant categories, payment methods, device
            activities and transaction patterns.

            <br><br>

            It provides fraud prediction, transaction investigation,
            exploratory data analysis, model insights, fraud analytics
            and historical trend analysis.

        </div>

    </div>
    """, unsafe_allow_html=True)

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

    st.subheader("📁 Connected Datasets")

    if datasets:

        rows = []

        for name, df in datasets.items():

            rows.append({
                "Dataset": name,
                "Transactions": len(df),
                "Features": df.shape[1],
                "Fraud Cases": fraud_count(df),
                "Missing Values": int(df.isnull().sum().sum())
            })

        show_table(pd.DataFrame(rows), len(rows))

    else:
        st.warning("Datasets not found. Check your data folders.")


# =========================================================
# 7. EXECUTIVE DASHBOARD - LIGHT DESIGN
# =========================================================

elif page == "📊 Executive Dashboard":

    st.markdown('<div class="light-dashboard">', unsafe_allow_html=True)

    st.title("🛡️ FraudShield AI")

    st.subheader(
        "Enterprise Fraud Detection & Risk Intelligence Platform"
    )

    st.caption(
        "Real-Time Monitoring • Machine Learning • Risk Scoring • "
        "Fraud Investigation • Graph Analytics"
    )

    st.divider()

    if datasets:

        selected = st.selectbox(
            "Select Transaction Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        target = get_target(df)
        amount_col = get_amount(df)
        date_col = get_date(df)

        total = len(df)
        fraud = fraud_count(df)
        normal = total - fraud
        rate = fraud / total * 100 if total else 0

        risk_col = next(
            (
                c for c in [
                    "Risk_Score",
                    "risk_score",
                    "RiskScore"
                ]
                if c in df.columns
            ),
            None
        )

        avg_risk = "Not Available"

        if risk_col:

            risk_values = pd.to_numeric(
                df[risk_col],
                errors="coerce"
            ).dropna()

            if not risk_values.empty:
                avg_risk = f"{risk_values.mean():.2f}"

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("📄 Transactions", f"{total:,}")
        k2.metric("🚨 Fraud Cases", f"{fraud:,}")
        k3.metric("⚠️ Fraud Rate", f"{rate:.2f}%")
        k4.metric("🎯 Avg Risk Score", avg_risk)

        st.divider()

        st.subheader("📊 Transaction Intelligence")

        left, right = st.columns(2)

        with left:

            if target:

                fig = px.pie(
                    names=["Normal Transactions", "Fraud Cases"],
                    values=[normal, fraud],
                    hole=0.55,
                    title="Fraud vs Normal Transactions",
                    color_discrete_sequence=[
                        "#269b78",
                        "#e85d5d"
                    ]
                )

                st.plotly_chart(fig, use_container_width=True)

        with right:

            if amount_col:

                fig = px.histogram(
                    df,
                    x=amount_col,
                    nbins=35,
                    title="Transaction Amount Distribution",
                    color_discrete_sequence=["#3185ce"]
                )

                st.plotly_chart(fig, use_container_width=True)

        if date_col and target:

            st.subheader("📈 Fraud Transaction Trend")

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

            if not trend_df.empty:

                trend = (
                    trend_df
                    .set_index(date_col)
                    .resample("D")[target]
                    .sum()
                    .reset_index()
                )

                trend.columns = [
                    "Date",
                    "Fraud Transactions"
                ]

                fig = px.line(
                    trend,
                    x="Date",
                    y="Fraud Transactions",
                    markers=True,
                    title="Daily Fraud Transactions",
                    color_discrete_sequence=["#277ac0"]
                )

                st.plotly_chart(fig, use_container_width=True)

        category_col = next(
            (
                c for c in [
                    "Merchant_Category",
                    "TransactionType",
                    "Payment_Method",
                    "Device_Type",
                    "Location"
                ]
                if c in df.columns
            ),
            None
        )

        if category_col and target:

            fraud_df = df[
                pd.to_numeric(df[target], errors="coerce")
                .fillna(0) == 1
            ]

            category_data = (
                fraud_df[category_col]
                .value_counts()
                .head(15)
                .reset_index()
            )

            category_data.columns = [
                category_col,
                "Fraud Cases"
            ]

            fig = px.bar(
                category_data,
                x=category_col,
                y="Fraud Cases",
                title=f"Fraud Cases by {category_col}",
                color_discrete_sequence=["#3185ce"]
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("📋 Transaction Records")

        fraud_only = st.checkbox("Show only fraud transactions")

        display_df = df.copy()

        if fraud_only and target:

            display_df = display_df[
                pd.to_numeric(display_df[target], errors="coerce")
                .fillna(0) == 1
            ]

        show_table(display_df, 200)

        st.download_button(
            "⬇️ Download Transaction Data",
            display_df.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_dashboard_data.csv",
            mime="text/csv"
        )

    else:
        st.error("No datasets found.")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 8. DATASET OVERVIEW
# =========================================================

elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Dataset Preview")
        show_table(df)

        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        show_table(info, len(info))

    else:
        st.warning("Datasets not found.")


# =========================================================
# 9. EDA DASHBOARD
# =========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        numeric = df.select_dtypes(include=np.number)

        st.subheader("Numerical Summary")

        if not numeric.empty:

            show_table(
                numeric.describe().T.reset_index(),
                100
            )

            st.subheader("Correlation Heatmap")

            fig = px.imshow(
                numeric.corr(),
                text_auto=True,
                aspect="auto",
                title="Feature Correlation"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Missing Values")

        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Datasets not found.")


# =========================================================
# 10. FEATURE ENGINEERING
# =========================================================

elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        numeric = df.select_dtypes(include=np.number).columns.tolist()

        categorical = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Features", df.shape[1])
        c2.metric("Numerical Features", len(numeric))
        c3.metric("Categorical Features", len(categorical))

        st.subheader("Numerical Features")
        st.write(numeric)

        st.subheader("Categorical Features")
        st.write(categorical)

        st.info(
            "This page describes existing dataset features. "
            "It does not change the original training data."
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 11. FRAUD INTELLIGENCE
# =========================================================

elif page == "🕵️ Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        target = get_target(df)

        if target:

            fraud_df = df[
                pd.to_numeric(df[target], errors="coerce")
                .fillna(0) == 1
            ]

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Cases", len(fraud_df))
            c2.metric("Normal Cases", len(df) - len(fraud_df))

            c3.metric(
                "Fraud Rate",
                f"{len(fraud_df) / len(df) * 100:.2f}%"
            )

            st.subheader("Fraud Transaction Records")
            show_table(fraud_df)

            amount_col = get_amount(df)

            if amount_col:

                fig = px.histogram(
                    fraud_df,
                    x=amount_col,
                    nbins=30,
                    title="Fraud Transaction Amount"
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 12. MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    files = [
        "credit_card_fraud_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]

    rows = []

    for filename in files:

        saved = load_model(filename)

        if saved is None:
            continue

        model = (
            saved.get("model")
            if isinstance(saved, dict)
            else saved
        )

        rows.append({
            "Model File": filename,
            "Algorithm": type(model).__name__,
            "Threshold": (
                saved.get("threshold")
                if isinstance(saved, dict)
                else "Not Available"
            ),
            "Features": (
                len(saved.get("features", []))
                if isinstance(saved, dict)
                else "Not Available"
            )
        })

    if rows:

        st.metric("Available Model Files", len(rows))
        show_table(pd.DataFrame(rows), len(rows))

    else:
        st.warning("No model artifacts found.")


# =========================================================
# 13. FRAUD PREDICTION
# =========================================================

elif page == "🚨 Fraud Prediction":

    st.title("🚨 Fraud Prediction")

    model_files = {
        "Credit Card Fraud": "credit_card_fraud_model.pkl",
        "Credit Card Fraud SMOTE": "credit_card_fraud_smote_model.pkl",
        "Fraud Detection": "fraud_detection_model.pkl",
        "Fraud Detection SMOTE": "fraud_detection_smote_model.pkl"
    }

    selected = st.selectbox("Choose Model", list(model_files.keys()))

    saved = load_model(model_files[selected])

    if saved is None:

        st.error("Model not found.")

    elif not isinstance(saved, dict):

        st.error("Model artifact format is not supported.")

    else:

        model = saved.get("model")
        threshold = saved.get("threshold", 0.5)
        features = saved.get("features", [])

        if model is None or not features:

            st.error("Model or feature list is missing.")

        else:

            st.write("Model:", type(model).__name__)
            st.write("Threshold:", threshold)

            with st.form("prediction_form"):

                values = {}
                cols = st.columns(2)

                for i, feature in enumerate(features):

                    with cols[i % 2]:

                        low = feature.lower()

                        if any(
                            x in low for x in [
                                "date", "time", "customer_id",
                                "transaction_id", "merchant",
                                "location", "device", "payment",
                                "category", "keyword", "type"
                            ]
                        ):

                            values[feature] = st.text_input(feature)

                        else:

                            values[feature] = st.number_input(
                                feature,
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

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                    c2.metric(
                        "Threshold",
                        f"{threshold * 100:.2f}%"
                    )

                    if probability >= threshold:
                        st.error("🚨 FRAUD TRANSACTION DETECTED")
                    else:
                        st.success("✅ NORMAL TRANSACTION")

                except Exception as error:
                    st.error(f"Prediction error: {error}")


# =========================================================
# 14. FRAUD ANALYTICS
# =========================================================

elif page == "📊 Fraud Analytics":

    st.title("📊 Fraud Analytics")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        target = get_target(df)

        if target:

            fraud_df = df[
                pd.to_numeric(df[target], errors="coerce")
                .fillna(0) == 1
            ]

            normal = len(df) - len(fraud_df)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Transactions", len(fraud_df))
            c2.metric("Normal Transactions", normal)
            c3.metric(
                "Fraud Percentage",
                f"{len(fraud_df) / len(df) * 100:.2f}%"
            )

            amount_col = get_amount(df)

            if amount_col:

                fig = px.box(
                    df,
                    x=target,
                    y=amount_col,
                    title="Transaction Amount by Fraud Class"
                )

                st.plotly_chart(fig, use_container_width=True)

            if "Location" in df.columns:

                location_data = (
                    fraud_df["Location"]
                    .value_counts()
                    .head(15)
                    .reset_index()
                )

                location_data.columns = ["Location", "Fraud Count"]

                fig = px.bar(
                    location_data,
                    x="Location",
                    y="Fraud Count",
                    title="Fraud by Location"
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 15. DETAILED INVESTIGATION
# =========================================================

elif page == "🔎 Detailed Investigation":

    st.title("🔎 Detailed Investigation")

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        target = get_target(df)
        amount_col = get_amount(df)

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

        search = st.text_input("Search Transaction or Customer ID")

        c1, c2 = st.columns(2)

        with c1:
            fraud_only = st.checkbox("Fraud only")

        with c2:
            high_amount = st.checkbox("Amount greater than 5000")

        filtered = df.copy()

        if search:

            if search_col == "All Columns":

                mask = filtered.astype(str).apply(
                    lambda col: col.str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

            else:

                filtered = filtered[
                    filtered[search_col].astype(str).str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ]

        if fraud_only and target:

            filtered = filtered[
                pd.to_numeric(filtered[target], errors="coerce")
                .fillna(0) == 1
            ]

        if high_amount and amount_col:

            filtered = filtered[
                pd.to_numeric(filtered[amount_col], errors="coerce") > 5000
            ]

        c1, c2, c3 = st.columns(3)

        c1.metric("Matching Records", len(filtered))

        if target:
            c2.metric("Fraud Records", fraud_count(filtered))

        if amount_col:
            c3.metric(
                "Total Amount",
                f"{pd.to_numeric(filtered[amount_col], errors='coerce').sum():,.2f}"
            )

        show_table(filtered)

        st.download_button(
            "⬇️ Download Results",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="investigation_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 16. EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    filename = st.selectbox(
        "Select Model",
        [
            "credit_card_fraud_model.pkl",
            "fraud_detection_model.pkl",
            "credit_card_fraud_smote_model.pkl",
            "fraud_detection_smote_model.pkl"
        ]
    )

    saved = load_model(filename)

    if saved is None or not isinstance(saved, dict):

        st.warning("Model artifact not found or unsupported.")

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
                title="Feature Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_table(importance, len(importance))

        else:
            st.info("Feature importance is not available for this model.")


# =========================================================
# 17. REAL-TIME MONITORING
# =========================================================

elif page == "📡 Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    st.info(
        "This page analyzes the available historical dataset. "
        "It is not connected to a live transaction stream."
    )

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        target = get_target(df)

        if target:

            fraud = fraud_count(df)
            normal = len(df) - fraud

            c1, c2, c3 = st.columns(3)

            c1.metric("Transactions", len(df))
            c2.metric("Fraud", fraud)
            c3.metric("Normal", normal)

            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[normal, fraud],
                hole=0.5,
                title="Transaction Status"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_table(df.tail(50), 50)

        else:
            st.warning("Fraud target not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 18. FRAUD FORECASTING
# =========================================================

elif page == "🔮 Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")

    st.info(
        "Historical fraud trend analysis. This page does not generate "
        "future predictions without a connected forecasting model."
    )

    if datasets:

        selected = st.selectbox("Choose Dataset", list(datasets.keys()))
        df = datasets[selected]

        date_col = get_date(df)
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

            trend_df = trend_df.dropna()

            if not trend_df.empty:

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
                    title="Historical Daily Fraud Transactions"
                )

                st.plotly_chart(fig, use_container_width=True)

                show_table(trend, 100)

        else:
            st.warning("Date column or fraud target not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 19. GRAPH ANALYTICS
# =========================================================

elif page == "🔗 Graph Analytics":

    st.title("🔗 Graph Analytics")

    st.info(
        "Relationship analysis using available transaction columns. "
        "This is not a dedicated graph neural network."
    )

    if fraud_df is not None:

        columns = [
            c for c in [
                "Customer_ID",
                "Merchant_Category",
                "Payment_Method",
                "Device_Type",
                "Location"
            ]
            if c in fraud_df.columns
        ]

        if columns:

            selected = st.selectbox(
                "Choose Relationship Feature",
                columns
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
                title=f"Transactions by {selected}"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_table(counts, 20)

        else:
            st.warning("Relationship columns not found.")

    else:
        st.warning("Fraud Detection Dataset not found.")


# =========================================================
# 20. FOOTER
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
