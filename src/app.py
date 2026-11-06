
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# app.py is inside src/, so parent.parent is project root
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================================================
# 2. DATASET AND MODEL FUNCTIONS
# =========================================================

@st.cache_data(show_spinner=False)
def read_csv_file(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_csv(filename):
    locations = [
        DATA_RAW / filename,
        DATA_PROCESSED / filename,
        BASE_DIR / filename
    ]

    for path in locations:
        if path.exists():
            return path

    for folder in [DATA_RAW, DATA_PROCESSED]:
        if folder.exists():
            matches = list(folder.rglob(filename))
            if matches:
                return matches[0]

    return None


def load_dataset(filename):
    path = find_csv(filename)

    if path:
        return read_csv_file(str(path))

    return None


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def get_model_object(saved):
    if isinstance(saved, dict):
        return saved.get("model")

    return saved


def get_model_features(saved, model):
    if isinstance(saved, dict) and saved.get("features"):
        return list(saved["features"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return []


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


def get_fraud_count(df):
    target = get_target(df)

    if target is None:
        return 0

    values = pd.to_numeric(
        df[target],
        errors="coerce"
    ).fillna(0)

    return int((values == 1).sum())


def get_normal_count(df):
    return max(0, len(df) - get_fraud_count(df))


def show_table(df, height=None):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height
    )


# =========================================================
# 3. LOAD YOUR ACTUAL DATASETS
# =========================================================

credit_df = load_dataset("credit_card_fraud_dataset.csv")
fraud_df = load_dataset("fraud_detection_dataset.csv")

datasets = {}

if credit_df is not None:
    datasets["Credit Card Fraud Dataset"] = credit_df

if fraud_df is not None:
    datasets["Fraud Detection Dataset"] = fraud_df


# =========================================================
# 4. GLOBAL CSS
# =========================================================

st.markdown("""
<style>

/* GENERAL */

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.stApp {
    transition: background 0.2s ease;
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

[data-testid="stSidebar"] label {
    color: #dbeafe;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 5px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 10px;
    border-radius: 9px;
    transition: 0.2s;
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
    margin-bottom: 28px;
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

/* HOMEPAGE PROJECT CARD */

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

/* LIGHT DASHBOARD */

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

/* METRIC CARDS */

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

/* FOOTER */

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
        letter-spacing:1px;
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
    "app",
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
# 6. HOMEPAGE
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
        get_fraud_count(df) for df in datasets.values()
    )

    model_files = list(MODEL_DIR.glob("*.pkl")) if MODEL_DIR.exists() else []

    notebook_files = list(BASE_DIR.rglob("*.ipynb"))

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    k2.metric(
        "Fraud Cases",
        f"{total_fraud:,}"
    )

    k3.metric(
        "ML Models",
        len(model_files)
    )

    k4.metric(
        "Notebooks",
        len(notebook_files)
    )

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
                "Fraud Cases": get_fraud_count(df),
                "Missing Values": int(df.isnull().sum().sum())
            })

        show_table(pd.DataFrame(rows))

    else:
        st.warning(
            "Datasets not found. Please check data/raw folder."
        )


# =========================================================
# 7. EXECUTIVE DASHBOARD
# =========================================================

elif page == "📊 Executive Dashboard":

    st.markdown(
        '<div class="light-dashboard">',
        unsafe_allow_html=True
    )

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
        amount_col = get_amount_column(df)
        date_col = get_date_column(df)

        total = len(df)
        fraud = get_fraud_count(df)
        normal = total - fraud

        rate = (
            fraud / total * 100
            if total > 0 else 0
        )

        risk_col = next(
            (
                col for col in [
                    "Risk_Score",
                    "risk_score",
                    "RiskScore"
                ]
                if col in df.columns
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

        k1.metric(
            "📄 Transactions",
            f"{total:,}"
        )

        k2.metric(
            "🚨 Fraud Cases",
            f"{fraud:,}"
        )

        k3.metric(
            "⚠️ Fraud Rate",
            f"{rate:.2f}%"
        )

        k4.metric(
            "🎯 Avg Risk Score",
            avg_risk
        )

        st.divider()

        st.subheader("📊 Transaction Intelligence")

        left, right = st.columns(2)

        with left:

            if target:

                fig = px.pie(
                    names=[
                        "Normal Transactions",
                        "Fraud Cases"
                    ],
                    values=[
                        normal,
                        fraud
                    ],
                    hole=0.55,
                    title="Fraud vs Normal Transactions",
                    color_discrete_sequence=[
                        "#269b78",
                        "#e85d5d"
                    ]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("Fraud target column not found.")

        with right:

            if amount_col:

                fig = px.histogram(
                    df,
                    x=amount_col,
                    nbins=35,
                    title="Transaction Amount Distribution",
                    color_discrete_sequence=["#3185ce"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("Transaction amount column not found.")

        # DAILY FRAUD TREND

        if date_col and target:

            st.subheader("📈 Fraud Transaction Trend")

            trend_df = df[
                [date_col, target]
            ].copy()

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

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # CATEGORY ANALYSIS

        category_col = next(
            (
                col for col in [
                    "Merchant_Category",
                    "TransactionType",
                    "Payment_Method",
                    "Device_Type",
                    "Location"
                ]
                if col in df.columns
            ),
            None
        )

        if category_col and target:

            st.subheader(
                f"📍 Fraud Analysis by {category_col}"
            )

            fraud_df = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
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

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # TRANSACTION TABLE

        st.divider()

        st.subheader("📋 Transaction Records")

        fraud_only = st.checkbox(
            "Show only fraud transactions"
        )

        display_df = df.copy()

        if fraud_only and target:

            display_df = display_df[
                pd.to_numeric(
                    display_df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

        show_table(display_df.head(200))

        st.download_button(
            "⬇️ Download Transaction Data",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_dashboard_data.csv",
            mime="text/csv"
        )

    else:

        st.error(
            "No datasets found in data/raw or data/processed."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 8. DATASET OVERVIEW
# =========================================================

elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", df.shape[1])
        c3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )
        c4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        st.subheader("Dataset Preview")
        show_table(df.head(200))

        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        show_table(info)

        st.subheader("Statistical Summary")

        show_table(
            df.describe(include="all").T.reset_index()
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 9. EDA DASHBOARD
# =========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        numeric = df.select_dtypes(
            include=np.number
        )

        categorical = df.select_dtypes(
            include=["object", "category"]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Numerical Columns", len(numeric.columns))
        c2.metric("Categorical Columns", len(categorical.columns))
        c3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader("Numerical Analysis")

        if not numeric.empty:

            show_table(
                numeric.describe().T.reset_index()
            )

            st.subheader("Correlation Heatmap")

            fig = px.imshow(
                numeric.corr(),
                text_auto=True,
                aspect="auto",
                title="Feature Correlation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Missing Values Analysis")

        missing = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing.columns = [
            "Column",
            "Missing Values"
        ]

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if len(categorical.columns) > 0:

            st.subheader("Categorical Analysis")

            selected_col = st.selectbox(
                "Select Categorical Feature",
                categorical.columns
            )

            counts = (
                df[selected_col]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                selected_col,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=selected_col,
                y="Count",
                title=f"Distribution of {selected_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 10. FEATURE ENGINEERING
# =========================================================

elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

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

        st.subheader("Feature Data Types")

        feature_info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values
        })

        show_table(feature_info)

        st.info(
            "This page inspects the existing dataset features. "
            "It does not modify your original training dataset."
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 11. FRAUD INTELLIGENCE
# =========================================================

elif page == "🕵️ Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]
        target = get_target(df)

        if target:

            fraud_data = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

            normal_count = len(df) - len(fraud_data)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Cases", len(fraud_data))
            c2.metric("Normal Cases", normal_count)

            c3.metric(
                "Fraud Rate",
                f"{len(fraud_data) / len(df) * 100:.2f}%"
            )

            amount_col = get_amount_column(df)

            if amount_col:

                fig = px.histogram(
                    fraud_data,
                    x=amount_col,
                    nbins=30,
                    title="Fraud Transaction Amount Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("Fraud Transaction Records")
            show_table(fraud_data.head(200))

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 12. MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    model_files = [
        "credit_card_fraud_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]

    rows = []

    for filename in model_files:

        saved = load_model(filename)

        if saved is None:
            continue

        model = get_model_object(saved)

        if model is None:
            continue

        threshold = (
            saved.get("threshold", "Not Available")
            if isinstance(saved, dict)
            else "Not Available"
        )

        features = get_model_features(saved, model)

        rows.append({
            "Model File": filename,
            "Algorithm": type(model).__name__,
            "Threshold": threshold,
            "Number of Features": len(features)
        })

    if rows:

        st.metric(
            "Available Model Files",
            len(rows)
        )

        show_table(pd.DataFrame(rows))

    else:

        st.warning(
            "No model files found in the models folder."
        )


# =========================================================
# 13. FRAUD PREDICTION
# =========================================================

elif page == "🚨 Fraud Prediction":

    st.title("🚨 Fraud Prediction")

    model_options = {
        "Credit Card Fraud": "credit_card_fraud_model.pkl",
        "Credit Card Fraud SMOTE": "credit_card_fraud_smote_model.pkl",
        "Fraud Detection": "fraud_detection_model.pkl",
        "Fraud Detection SMOTE": "fraud_detection_smote_model.pkl"
    }

    selected = st.selectbox(
        "Select Prediction Model",
        list(model_options.keys())
    )

    saved = load_model(model_options[selected])

    if saved is None:

        st.error("Selected model was not found.")

    elif not isinstance(saved, dict):

        st.error(
            "This model file does not contain the expected "
            "model, threshold and feature information."
        )

    else:

        model = get_model_object(saved)
        features = get_model_features(saved, model)
        threshold = saved.get("threshold", 0.5)

        if model is None or not features:

            st.error(
                "Model or saved feature list is missing."
            )

        else:

            st.write(
                f"**Algorithm:** {type(model).__name__}"
            )

            st.write(
                f"**Saved threshold:** {threshold}"
            )

            st.caption(
                "Enter values matching the model's original training features."
            )

            with st.form("fraud_prediction_form"):

                values = {}

                columns = st.columns(2)

                for i, feature in enumerate(features):

                    with columns[i % 2]:

                        feature_lower = feature.lower()

                        categorical_words = [
                            "location",
                            "merchant_category",
                            "transactiontype",
                            "payment_method",
                            "device_type",
                            "suspicious_keyword",
                            "channel",
                            "promo_id"
                        ]

                        is_categorical = any(
                            word in feature_lower
                            for word in categorical_words
                        )

                        if is_categorical:

                            values[feature] = st.text_input(
                                feature
                            )

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

                    input_df = pd.DataFrame(
                        [values],
                        columns=features
                    )

                    if hasattr(model, "feature_names_in_"):
                        input_df = input_df.reindex(
                            columns=model.feature_names_in_
                        )

                    probability = model.predict_proba(
                        input_df
                    )[0][1]

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                    c2.metric(
                        "Model Threshold",
                        f"{float(threshold) * 100:.2f}%"
                    )

                    if probability >= float(threshold):

                        st.error(
                            "🚨 FRAUD TRANSACTION DETECTED!"
                        )

                    else:

                        st.success(
                            "✅ NORMAL TRANSACTION"
                        )

                except Exception as error:

                    st.error(
                        f"Prediction error: {error}"
                    )

                    st.info(
                        "Check whether the input features and their "
                        "data types match the model training pipeline."
                    )


# =========================================================
# 14. FRAUD ANALYTICS
# =========================================================

elif page == "📊 Fraud Analytics":

    st.title("📊 Fraud Analytics")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]
        target = get_target(df)

        if target:

            fraud_data = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

            normal = len(df) - len(fraud_data)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Transactions", len(fraud_data))
            c2.metric("Normal Transactions", normal)

            c3.metric(
                "Fraud Percentage",
                f"{len(fraud_data) / len(df) * 100:.2f}%"
            )

            amount_col = get_amount_column(df)

            if amount_col:

                fig = px.box(
                    df,
                    x=target,
                    y=amount_col,
                    title="Transaction Amount by Fraud Class"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            if "Location" in df.columns:

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
                    title="Fraud by Location"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Datasets not found.")


# =========================================================
# 15. DETAILED INVESTIGATION
# =========================================================

elif page == "🔎 Detailed Investigation":

    st.title("🔎 Detailed Investigation")

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        target = get_target(df)
        amount_col = get_amount_column(df)

        search_columns = [
            col for col in [
                "TransactionID",
                "Transaction_ID",
                "Customer_ID",
                "receipt_id"
            ]
            if col in df.columns
        ]

        search_options = ["All Columns"] + search_columns

        search_col = st.selectbox(
            "Search By",
            search_options
        )

        search_text = st.text_input(
            "Enter Transaction ID or Customer ID"
        )

        c1, c2 = st.columns(2)

        with c1:
            fraud_only = st.checkbox("Fraud Only")

        with c2:
            high_amount = st.checkbox(
                "Amount Greater Than 5000"
            )

        filtered = df.copy()

        if search_text:

            if search_col == "All Columns":

                mask = filtered.astype(str).apply(
                    lambda col: col.str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

            else:

                filtered = filtered[
                    filtered[search_col].astype(str).str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ]

        if fraud_only and target:

            filtered = filtered[
                pd.to_numeric(
                    filtered[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

        if high_amount and amount_col:

            filtered = filtered[
                pd.to_numeric(
                    filtered[amount_col],
                    errors="coerce"
                ) > 5000
            ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Matching Records",
            len(filtered)
        )

        if target:
            c2.metric(
                "Fraud Records",
                get_fraud_count(filtered)
            )

        if amount_col:

            amount_total = pd.to_numeric(
                filtered[amount_col],
                errors="coerce"
            ).sum()

            c3.metric(
                "Total Amount",
                f"{amount_total:,.2f}"
            )

        st.subheader("Investigation Results")

        show_table(filtered.head(500))

        st.download_button(
            "⬇️ Download Investigation Results",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_investigation.csv",
            mime="text/csv"
        )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 16. EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    model_files = [
        "credit_card_fraud_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]

    selected = st.selectbox(
        "Select Model",
        model_files
    )

    saved = load_model(selected)

    if saved is None:

        st.warning("Model file not found.")

    else:

        model = get_model_object(saved)
        features = get_model_features(saved, model)

        if hasattr(model, "feature_importances_"):

            importances = model.feature_importances_

            if len(features) == len(importances):

                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                })

                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=False
                )

                fig = px.bar(
                    importance_df.head(20).sort_values(
                        "Importance"
                    ),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 20 Feature Importances"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                show_table(importance_df)

            else:

                st.warning(
                    "Feature names do not match the model's importance values."
                )

        else:

            st.info(
                "Feature importance is not directly available for this model."
            )


# =========================================================
# 17. REAL-TIME MONITORING
# =========================================================

elif page == "📡 Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    st.info(
        "This page displays monitoring insights from your available "
        "historical dataset. It is not connected to a live transaction stream."
    )

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]
        target = get_target(df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Transactions",
            f"{len(df):,}"
        )

        if target:

            fraud = get_fraud_count(df)
            normal = len(df) - fraud

            c2.metric("Fraud Transactions", fraud)
            c3.metric("Normal Transactions", normal)

            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[normal, fraud],
                hole=0.5,
                title="Transaction Status"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Latest Available Records")
        show_table(df.tail(50))

    else:
        st.warning("Datasets not found.")


# =========================================================
# 18. FRAUD FORECASTING
# =========================================================

elif page == "🔮 Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")

    st.info(
        "This page presents historical fraud trends. "
        "Future forecasts require a separately trained forecasting model."
    )

    if datasets:

        selected = st.selectbox(
            "Select Dataset",
            list(datasets.keys())
        )

        df = datasets[selected]

        date_col = get_date_column(df)
        target = get_target(df)

        if date_col and target:

            trend_df = df[
                [date_col, target]
            ].copy()

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
                    title="Historical Daily Fraud Transactions"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                show_table(trend)

            else:

                st.warning(
                    "No valid date records found."
                )

        else:

            st.warning(
                "Date column or fraud target column not found."
            )

    else:
        st.warning("Datasets not found.")


# =========================================================
# 19. GRAPH ANALYTICS
# =========================================================

elif page == "🔗 Graph Analytics":

    st.title("🔗 Graph Analytics")

    st.info(
        "This page explores relationships in the available transaction "
        "data. It is not a dedicated graph neural network."
    )

    if fraud_df is not None:

        relationship_columns = [
            col for col in [
                "Customer_ID",
                "Merchant_Category",
                "Payment_Method",
                "Device_Type",
                "Location"
            ]
            if col in fraud_df.columns
        ]

        if relationship_columns:

            selected_col = st.selectbox(
                "Select Relationship Feature",
                relationship_columns
            )

            counts = (
                fraud_df[selected_col]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                selected_col,
                "Transaction Count"
            ]

            fig = px.bar(
                counts,
                x=selected_col,
                y="Transaction Count",
                title=f"Transaction Relationships by {selected_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            show_table(counts)

        else:

            st.warning(
                "No supported relationship columns found."
            )

    else:

        st.warning(
            "Fraud Detection Dataset not found."
        )


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
