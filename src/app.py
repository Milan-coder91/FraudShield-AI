
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import textwrap

from pathlib import Path

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# ==========================================================
# 2. DARK THEME CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #080f1b;
    color: #eaf1fa;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020817, #10223a);
    border-right: 1px solid #263c56;
}

[data-testid="stSidebar"] * {
    color: #e4efff;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 5px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px 10px;
    border-radius: 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1a3554;
}

.hero {
    background: linear-gradient(115deg, #0799e8, #1761e8 58%, #174bd0);
    padding: 45px 42px;
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(0, 80, 200, 0.23);
}

.hero-title {
    font-size: 46px;
    font-weight: 850;
    color: white;
}

.hero-subtitle {
    font-size: 23px;
    font-weight: 700;
    color: white;
    margin-top: 18px;
}

.hero-description {
    color: #e1efff;
    font-size: 15px;
    margin-top: 20px;
    line-height: 1.8;
}

.project-card {
    background: linear-gradient(145deg, #111e30, #0d1929);
    border: 1px solid #243c57;
    border-radius: 16px;
    padding: 25px;
    margin: 22px 0;
}

.project-heading {
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 15px;
}

.project-text {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.9;
}

.home-kpi {
    background: #101d2d;
    border: 1px solid #263e59;
    padding: 20px;
    border-radius: 13px;
    min-height: 105px;
}

.home-kpi-label {
    color: #9fb2c9;
    font-size: 14px;
}

.home-kpi-value {
    color: white;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.pipeline-container,
.feature-container {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    overflow-x: auto;
    padding: 14px 2px;
}

.pipeline-step,
.feature-item {
    flex: 1 0 auto;
    min-width: 120px;
    white-space: nowrap;
    text-align: center;
    background: #142c45;
    border: 1px solid #315579;
    border-radius: 10px;
    padding: 15px 10px;
    color: #eaf4ff;
    font-size: 13px;
    font-weight: 650;
}

.pipeline-arrow {
    color: #38bdf8;
    font-size: 20px;
}

.footer {
    text-align: center;
    color: #8295ac;
    padding: 25px;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #101d2d;
    border: 1px solid #263e59;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #a9bdd5;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stMultiSelect label {
    color: #eaf1fa !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. HTML HELPER
# ==========================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ==========================================================
# 4. CHART DESIGN HELPER
# ==========================================================

def show_chart(fig, height=420):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#080f1b",
        plot_bgcolor="#101d2d",
        font=dict(
            family="Arial",
            color="#eaf1fa",
            size=13
        ),
        title=dict(
            font=dict(size=20, color="#ffffff"),
            x=0.02
        ),
        margin=dict(
            l=35,
            r=25,
            t=65,
            b=45
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf1fa")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#26394f",
            zerolinecolor="#26394f"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#26394f",
            zerolinecolor="#26394f"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None
    )


# ==========================================================
# 5. DATASET LOADING
# ==========================================================

@st.cache_data(show_spinner=False)
def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_file(filename):

    folders = [
        RAW_DIR,
        PROCESSED_DIR,
        DATA_DIR,
        BASE_DIR
    ]

    for folder in folders:

        candidate = folder / filename

        if candidate.exists():
            return candidate

    for folder in folders:

        if folder.exists():

            matches = list(folder.rglob(filename))

            if matches:
                return matches[0]

    return None


def load_dataset(filename):

    path = find_file(filename)

    if path is None:
        return None

    return read_csv(str(path))


credit_df = load_dataset("credit_card_fraud_dataset.csv")
fraud_df = load_dataset("fraud_detection_dataset.csv")

datasets = {}

if credit_df is not None:
    datasets["Credit Card Fraud Dataset"] = credit_df

if fraud_df is not None:
    datasets["Fraud Detection Dataset"] = fraud_df


# ==========================================================
# 6. COMMON DATA FUNCTIONS
# ==========================================================

def target_column(df):

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


def amount_column(df):

    for col in [
        "Amount",
        "Transaction_Amount",
        "total_value"
    ]:

        if col in df.columns:
            return col

    return None


def date_column(df):

    for col in [
        "TransactionDate",
        "Transaction_Date",
        "date"
    ]:

        if col in df.columns:
            return col

    return None


def fraud_count(df):

    col = target_column(df)

    if col is None:
        return 0

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

    return int((values == 1).sum())


def normal_count(df):

    return max(0, len(df) - fraud_count(df))


def model_files():

    if not MODEL_DIR.exists():
        return []

    return sorted(MODEL_DIR.glob("*.pkl"))


def load_saved_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def get_model(saved):

    if isinstance(saved, dict):
        return saved.get("model")

    return saved


def get_features(saved, model):

    if isinstance(saved, dict):

        if saved.get("features") is not None:
            return list(saved["features"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return []


def show_table(df, rows=200):

    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


def dataset_selector():

    return st.selectbox(
        "Select Dataset",
        list(datasets.keys())
    )


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

# Sidebar ke top par extra logo/branding intentionally nahi hai.

st.sidebar.markdown("## Navigation")

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
    "Select Page",
    pages,
    index=0,
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.caption("FraudShield AI | Fraud Intelligence Platform")


# ==========================================================
# 8. HOMEPAGE
# ==========================================================

if page == "🏠 app":

    render_html("""
    <div class="hero">
        <div class="hero-title">🛡️ FraudShield AI</div>

        <div class="hero-subtitle">
            Enterprise Fraud Detection &amp; Risk Intelligence Platform
        </div>

        <div class="hero-description">
            Real-Time Monitoring • Machine Learning • Risk Scoring •
            Fraud Investigation • Graph Analytics
        </div>
    </div>
    """)

    total_transactions = sum(
        len(df) for df in datasets.values()
    )

    total_fraud = sum(
        fraud_count(df) for df in datasets.values()
    )

    available_models = len(model_files())

    notebook_count = len(
        list(BASE_DIR.rglob("*.ipynb"))
    )

    k1, k2, k3, k4 = st.columns(4)

    metrics = [
        ("Transactions", total_transactions),
        ("Fraud Cases", total_fraud),
        ("ML Models", available_models),
        ("Notebooks", notebook_count)
    ]

    for col, (label, value) in zip(
        [k1, k2, k3, k4],
        metrics
    ):

        with col:

            render_html(f"""
            <div class="home-kpi">
                <div class="home-kpi-label">{label}</div>
                <div class="home-kpi-value">{value:,}</div>
            </div>
            """)

    render_html("""
    <div class="project-card">
        <div class="project-heading">🚀 Project Overview</div>

        <div class="project-text">
            FraudShield AI is an intelligent financial fraud detection
            platform designed to identify suspicious and potentially
            fraudulent transactions using Machine Learning.

            <br><br>

            The platform analyzes transaction behavior, customer
            information, merchant categories, payment methods,
            device activities and transaction patterns.

            <br><br>

            It provides fraud prediction, transaction investigation,
            exploratory data analysis, model insights and fraud analytics.
        </div>
    </div>
    """)

    st.subheader("⚡ AI Fraud Detection Pipeline")

    render_html("""
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
    """)

    st.divider()

    st.subheader("✨ Platform Features")

    render_html("""
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
    """)

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

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Datasets not found. Check data/raw folder.")


# ==========================================================
# 9. EXECUTIVE DASHBOARD
# ==========================================================

elif page == "📊 Executive Dashboard":

    st.title("🛡️ FraudShield AI")

    st.subheader(
        "Enterprise Fraud Detection & Risk Intelligence Platform"
    )

    st.caption(
        "Real-Time Monitoring • Machine Learning • Risk Scoring • "
        "Fraud Investigation • Graph Analytics"
    )

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        target = target_column(df)
        amount = amount_column(df)
        date = date_column(df)

        total = len(df)
        fraud = fraud_count(df)
        normal = normal_count(df)

        rate = (
            fraud / total * 100
            if total else 0
        )

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

        risk_value = "N/A"

        if risk_col:

            values = pd.to_numeric(
                df[risk_col],
                errors="coerce"
            ).dropna()

            if not values.empty:
                risk_value = f"{values.mean():.2f}"

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📄 Transactions", f"{total:,}")
        c2.metric("🚨 Fraud Cases", f"{fraud:,}")
        c3.metric("⚠️ Fraud Rate", f"{rate:.2f}%")
        c4.metric("🎯 Avg Risk Score", risk_value)

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
                        "#16a085",
                        "#e74c3c"
                    ]
                )

                show_chart(fig)

            else:

                st.info("Fraud target column not found.")

        with right:

            if amount:

                fig = px.histogram(
                    df,
                    x=amount,
                    nbins=35,
                    title="Transaction Amount Distribution",
                    color_discrete_sequence=["#3498db"]
                )

                show_chart(fig)

            else:

                st.info("Transaction amount column not found.")

        if date and target:

            st.subheader("📈 Fraud Transaction Trend")

            trend_df = df[[date, target]].copy()

            trend_df[date] = pd.to_datetime(
                trend_df[date],
                errors="coerce"
            )

            trend_df[target] = pd.to_numeric(
                trend_df[target],
                errors="coerce"
            )

            trend_df = trend_df.dropna(
                subset=[date, target]
            )

            if not trend_df.empty:

                trend = (
                    trend_df
                    .set_index(date)
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
                    title="Daily Fraud Transactions"
                )

                fig.update_traces(
                    line=dict(color="#38bdf8", width=3),
                    marker=dict(size=7)
                )

                show_chart(fig)

        category = next(
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

        if category and target:

            st.subheader(f"📍 Fraud Analysis by {category}")

            fraud_records = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

            category_data = (
                fraud_records[category]
                .value_counts()
                .head(15)
                .reset_index()
            )

            category_data.columns = [
                category,
                "Fraud Cases"
            ]

            fig = px.bar(
                category_data,
                x=category,
                y="Fraud Cases",
                title=f"Fraud Cases by {category}",
                color="Fraud Cases",
                color_continuous_scale="Blues"
            )

            show_chart(fig)

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

        show_table(display_df)

        st.download_button(
            "⬇️ Download Transaction Data",
            data=display_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="fraudshield_dashboard_data.csv",
            mime="text/csv"
        )

    else:

        st.error("No datasets found.")


# ==========================================================
# 10. DATASET OVERVIEW
# ==========================================================

elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", df.shape[1])
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

        st.dataframe(
            info,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 11. EDA DASHBOARD
# ==========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    if datasets:

        selected = dataset_selector()
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
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        if not numeric.empty:

            st.subheader("Numerical Analysis")

            st.dataframe(
                numeric.describe().T,
                use_container_width=True
            )

            st.subheader("Correlation Heatmap")

            correlation = numeric.corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Heatmap"
            )

            show_chart(fig, height=600)

            st.subheader("Numerical Feature Distribution")

            selected_num = st.selectbox(
                "Select Numerical Feature",
                numeric.columns
            )

            fig = px.histogram(
                df,
                x=selected_num,
                nbins=40,
                title=f"Distribution of {selected_num}",
                color_discrete_sequence=["#38bdf8"]
            )

            show_chart(fig)

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
            title="Missing Values by Column",
            color="Missing Values",
            color_continuous_scale="Blues"
        )

        show_chart(fig)

        if len(categorical.columns):

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
                title=f"Distribution of {selected_col}",
                color="Count",
                color_continuous_scale="Teal"
            )

            show_chart(fig)

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 12. FEATURE ENGINEERING
# ==========================================================

elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    if datasets:

        selected = dataset_selector()
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

        info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            info,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This page inspects existing dataset features. "
            "Original training data is not modified."
        )

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 13. FRAUD INTELLIGENCE
# ==========================================================

elif page == "🕵️ Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        target = target_column(df)

        if target:

            fraud_records = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

            normal = len(df) - len(fraud_records)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Cases", len(fraud_records))
            c2.metric("Normal Cases", normal)

            c3.metric(
                "Fraud Rate",
                f"{len(fraud_records) / len(df) * 100:.2f}%"
            )

            amount = amount_column(df)

            if amount:

                fig = px.histogram(
                    fraud_records,
                    x=amount,
                    nbins=30,
                    title="Fraud Transaction Amount Distribution",
                    color_discrete_sequence=["#ef5350"]
                )

                show_chart(fig)

            st.subheader("Fraud Transaction Records")
            show_table(fraud_records)

        else:

            st.warning("Fraud target column not found.")

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 14. MODEL COMPARISON
# ==========================================================

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

        saved = load_saved_model(filename)

        if saved is None:
            continue

        model = get_model(saved)

        if model is None:
            continue

        threshold = (
            saved.get("threshold", "Not Available")
            if isinstance(saved, dict)
            else "Not Available"
        )

        features = get_features(saved, model)

        rows.append({
            "Model File": filename,
            "Algorithm": type(model).__name__,
            "Threshold": threshold,
            "Number of Features": len(features)
        })

    if rows:

        st.metric("Available Models", len(rows))

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("No model files found in models folder.")


# ==========================================================
# 15. FRAUD PREDICTION
# ==========================================================

elif page == "🚨 Fraud Prediction":

    st.title("🚨 Fraud Prediction")

    options = {
        "Credit Card Fraud": "credit_card_fraud_model.pkl",
        "Credit Card Fraud SMOTE": "credit_card_fraud_smote_model.pkl",
        "Fraud Detection": "fraud_detection_model.pkl",
        "Fraud Detection SMOTE": "fraud_detection_smote_model.pkl"
    }

    selected = st.selectbox(
        "Select Prediction Model",
        list(options.keys())
    )

    saved = load_saved_model(options[selected])

    if saved is None:

        st.error("Selected model was not found.")

    elif not isinstance(saved, dict):

        st.error(
            "Expected saved model dictionary containing model, "
            "threshold and features."
        )

    else:

        model = get_model(saved)
        features = get_features(saved, model)
        threshold = saved.get("threshold", 0.5)

        if model is None or not features:

            st.error("Model or saved feature list is missing.")

        else:

            st.write("Algorithm:", type(model).__name__)
            st.write("Saved threshold:", threshold)

            st.caption(
                "Enter feature values using the same format "
                "as the original model training data."
            )

            with st.form("fraud_prediction_form"):

                values = {}
                columns = st.columns(2)

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

                for i, feature in enumerate(features):

                    with columns[i % 2]:

                        feature_lower = feature.lower()

                        if any(
                            word in feature_lower
                            for word in categorical_words
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

                        st.error("🚨 FRAUD TRANSACTION DETECTED!")

                    else:

                        st.success("✅ NORMAL TRANSACTION")

                except Exception as error:

                    st.error(f"Prediction error: {error}")

                    st.info(
                        "Check input data types and encoding against "
                        "the original model training pipeline."
                    )


# ==========================================================
# 16. FRAUD ANALYTICS
# ==========================================================

elif page == "📊 Fraud Analytics":

    st.title("📊 Fraud Analytics")

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        target = target_column(df)

        if target:

            fraud_records = df[
                pd.to_numeric(
                    df[target],
                    errors="coerce"
                ).fillna(0) == 1
            ]

            normal = len(df) - len(fraud_records)

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Transactions", len(fraud_records))
            c2.metric("Normal Transactions", normal)

            c3.metric(
                "Fraud Percentage",
                f"{len(fraud_records) / len(df) * 100:.2f}%"
            )

            amount = amount_column(df)

            if amount:

                fig = px.box(
                    df,
                    x=target,
                    y=amount,
                    title="Transaction Amount by Fraud Class",
                    color=target
                )

                show_chart(fig)

            if "Location" in df.columns:

                location_data = (
                    fraud_records["Location"]
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
                    title="Fraud by Location",
                    color="Fraud Count",
                    color_continuous_scale="Reds"
                )

                show_chart(fig)

        else:

            st.warning("Fraud target column not found.")

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 17. DETAILED INVESTIGATION
# ==========================================================

elif page == "🔎 Detailed Investigation":

    st.title("🔎 Detailed Investigation")

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        target = target_column(df)
        amount = amount_column(df)

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

        if high_amount and amount:

            filtered = filtered[
                pd.to_numeric(
                    filtered[amount],
                    errors="coerce"
                ) > 5000
            ]

        c1, c2, c3 = st.columns(3)

        c1.metric("Matching Records", len(filtered))

        if target:
            c2.metric("Fraud Records", fraud_count(filtered))

        if amount:

            total_amount = pd.to_numeric(
                filtered[amount],
                errors="coerce"
            ).sum()

            c3.metric(
                "Total Amount",
                f"{total_amount:,.2f}"
            )

        st.subheader("Investigation Results")

        show_table(filtered, rows=500)

        st.download_button(
            "⬇️ Download Investigation Results",
            data=filtered.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="fraudshield_investigation.csv",
            mime="text/csv"
        )

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 18. EXPLAINABLE AI
# ==========================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    files = [
        "credit_card_fraud_model.pkl",
        "credit_card_fraud_smote_model.pkl",
        "fraud_detection_model.pkl",
        "fraud_detection_smote_model.pkl"
    ]

    selected = st.selectbox(
        "Select Model",
        files
    )

    saved = load_saved_model(selected)

    if saved is None:

        st.warning("Model file not found.")

    else:

        model = get_model(saved)
        features = get_features(saved, model)

        if hasattr(model, "feature_importances_"):

            importance = model.feature_importances_

            if len(features) == len(importance):

                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values(
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
                    title="Top 20 Feature Importances",
                    color="Importance",
                    color_continuous_scale="Blues"
                )

                show_chart(fig, height=600)

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "Feature names do not match importance values."
                )

        else:

            st.info(
                "Feature importance is not directly available "
                "for this model."
            )


# ==========================================================
# 19. REAL-TIME MONITORING
# ==========================================================

elif page == "📡 Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    st.info(
        "This page displays monitoring insights from historical data. "
        "It is not connected to a live transaction stream."
    )

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        target = target_column(df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Transactions",
            f"{len(df):,}"
        )

        if target:

            fraud = fraud_count(df)
            normal = len(df) - fraud

            c2.metric("Fraud Transactions", fraud)
            c3.metric("Normal Transactions", normal)

            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[normal, fraud],
                hole=0.5,
                title="Transaction Status",
                color_discrete_sequence=[
                    "#16a085",
                    "#e74c3c"
                ]
            )

            show_chart(fig)

        st.subheader("Latest Available Records")

        st.dataframe(
            df.tail(50),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 20. FRAUD FORECASTING
# ==========================================================

elif page == "🔮 Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")

    st.info(
        "This page displays historical fraud trends. "
        "A future forecast requires a separately trained forecasting model."
    )

    if datasets:

        selected = dataset_selector()
        df = datasets[selected]

        date = date_column(df)
        target = target_column(df)

        if date and target:

            trend_df = df[[date, target]].copy()

            trend_df[date] = pd.to_datetime(
                trend_df[date],
                errors="coerce"
            )

            trend_df[target] = pd.to_numeric(
                trend_df[target],
                errors="coerce"
            )

            trend_df = trend_df.dropna(
                subset=[date, target]
            )

            if not trend_df.empty:

                trend = (
                    trend_df
                    .set_index(date)
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

                fig.update_traces(
                    line=dict(color="#38bdf8", width=3),
                    marker=dict(size=7)
                )

                show_chart(fig)

                st.dataframe(
                    trend,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning("No valid date records found.")

        else:

            st.warning(
                "Date column or fraud target column not found."
            )

    else:

        st.warning("Datasets not found.")


# ==========================================================
# 21. GRAPH ANALYTICS
# ==========================================================

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
                title=f"Transaction Relationships by {selected_col}",
                color="Transaction Count",
                color_continuous_scale="Teal"
            )

            show_chart(fig)

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No supported relationship columns found."
            )

    else:

        st.warning("Fraud Detection Dataset not found.")


# ==========================================================
# 22. FOOTER
# ==========================================================

render_html("""
<div class="footer">
    🛡️ <b>FraudShield AI</b>
    <br>
    Enterprise Fraud Detection &amp; Risk Intelligence Platform
    <br>
    Built with Python • Machine Learning • Streamlit
</div>
""")
