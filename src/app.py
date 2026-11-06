
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================

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

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #07111f;
    color: #e5edf7;
}

[data-testid="stSidebar"] {
    background-color: #0d1b2a;
    border-right: 1px solid #24364b;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

p, label, span {
    color: #cbd5e1;
}

div[data-testid="stMetric"] {
    background: #102238;
    border: 1px solid #263d56;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8;
}

.section-card {
    background: #102238;
    border: 1px solid #263d56;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
}

.pipeline-container {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    overflow-x: auto;
    padding: 12px 4px;
}

.pipeline-step {
    flex: 1 0 auto;
    min-width: 100px;
    white-space: nowrap;
    text-align: center;
    padding: 13px 10px;
    background: #142c45;
    border: 1px solid #2b5275;
    border-radius: 9px;
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 600;
}

.pipeline-arrow {
    flex: 0 0 auto;
    color: #38bdf8;
    font-size: 20px;
}

.feature-container {
    display: flex;
    flex-wrap: nowrap;
    align-items: center;
    gap: 12px;
    width: 100%;
    overflow-x: auto;
    padding: 10px 0;
}

.feature-item {
    flex: 1 0 auto;
    min-width: 120px;
    white-space: nowrap;
    text-align: center;
    background: #142c45;
    border: 1px solid #2b5275;
    padding: 13px 10px;
    border-radius: 9px;
    color: #e2e8f0;
    font-size: 13px;
}

div.stButton > button {
    background: #0284c7;
    color: white;
    border: none;
    border-radius: 8px;
}

div.stButton > button:hover {
    background: #0369a1;
    color: white;
}

hr {
    border-color: #263d56;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

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


def load_dataset(filename):
    path = find_dataset(filename)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Dataset loading error: {error}")
        return None


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


def get_target_column(df):
    candidates = [
        "IsFraud",
        "Fraudulent",
        "is_fraud",
        "fraud",
        "Class"
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def metric_card(label, value):
    st.metric(label, value)


def show_dataframe(df):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def show_model_info(filename):
    saved = load_model(filename)

    if saved is None:
        st.warning(f"Model not found: {filename}")
        return

    if isinstance(saved, dict):
        model = saved.get("model")
        threshold = saved.get("threshold")
        features = saved.get("features", [])

        st.write("**Model:**", type(model).__name__)
        st.write("**Threshold:**", threshold)
        st.write("**Number of features:**", len(features))
    else:
        st.write("**Model:**", type(saved).__name__)


# =====================================================
# LOAD DATASETS
# =====================================================

credit_df = load_dataset("credit_card_fraud_dataset.csv")
fraud_df = load_dataset("fraud_detection_dataset.csv")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("# 🛡️ FraudShield AI")
st.sidebar.caption("Intelligent Fraud Detection Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
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
)

st.sidebar.divider()
st.sidebar.caption("FraudShield AI | Machine Learning Platform")


# =====================================================
# HEADER
# =====================================================

st.title("🛡️ FraudShield AI")
st.caption("AI-Powered Financial Fraud Detection & Intelligence Platform")

st.divider()


# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

if page == "Executive Dashboard":

    st.subheader("Executive Dashboard")
    st.write("Overview of fraud detection performance and transaction intelligence.")

    if credit_df is not None:
        df = credit_df.copy()
        target = get_target_column(df)

        total = len(df)
        fraud_count = int(df[target].sum()) if target else 0
        normal_count = total - fraud_count
        fraud_rate = fraud_count / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraud Transactions", f"{fraud_count:,}")
        c3.metric("Normal Transactions", f"{normal_count:,}")
        c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.divider()

        left, right = st.columns(2)

        with left:
            if target:
                counts = df[target].value_counts().rename(
                    index={0: "Normal", 1: "Fraud"}
                )
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title="Fraud vs Normal Transactions",
                    hole=0.55
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            amount_col = next(
                (c for c in ["Amount", "Transaction_Amount", "total_value"]
                 if c in df.columns),
                None
            )

            if amount_col:
                fig = px.histogram(
                    df,
                    x=amount_col,
                    title="Transaction Amount Distribution",
                    nbins=40
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Credit Card dataset not found. Check your data folder.")

    st.divider()
    st.subheader("AI Fraud Detection Pipeline")

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
    st.subheader("Platform Features")

    st.markdown("""
    <div class="feature-container">
        <div class="feature-item">📊 Dashboard</div>
        <div class="feature-item">🔎 Investigation</div>
        <div class="feature-item">🤖 ML Prediction</div>
        <div class="feature-item">📈 Analytics</div>
        <div class="feature-item">🧠 Explainable AI</div>
        <div class="feature-item">📡 Monitoring</div>
        <div class="feature-item">🔗 Graph Analytics</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.subheader("Dataset Overview")

    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:
        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.subheader("Dataset Preview")
        show_dataframe(df.head(100))

        st.subheader("Column Information")
        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })
        show_dataframe(info)

    else:
        st.error("Selected dataset not found.")


# =====================================================
# EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.subheader("Exploratory Data Analysis")

    dataset_choice = st.selectbox(
        "Choose dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:

        numeric_df = df.select_dtypes(include=np.number)

        st.subheader("Numerical Summary")
        show_dataframe(numeric_df.describe().T.reset_index())

        st.subheader("Correlation Heatmap")

        if not numeric_df.empty:
            corr = numeric_df.corr()

            fig = px.imshow(
                corr,
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
        st.error("Dataset not found.")


# =====================================================
# FEATURE ENGINEERING
# =====================================================

elif page == "Feature Engineering":

    st.subheader("Feature Engineering")

    dataset_choice = st.selectbox(
        "Select dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:

        st.metric("Total Features", df.shape[1])

        feature_info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values
        })

        show_dataframe(feature_info)

        st.subheader("Numerical Features")

        numerical = df.select_dtypes(include=np.number).columns.tolist()
        st.write(numerical)

        st.subheader("Categorical Features")

        categorical = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        st.write(categorical)

    else:
        st.warning("Dataset not found.")


# =====================================================
# FRAUD INTELLIGENCE
# =====================================================

elif page == "Fraud Intelligence":

    st.subheader("Fraud Intelligence")

    dataset_choice = st.selectbox(
        "Select dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:

        target = get_target_column(df)

        if target:

            fraud_records = df[df[target] == 1]

            st.metric("Detected Fraud Records", f"{len(fraud_records):,}")

            st.subheader("Fraud Records")
            show_dataframe(fraud_records.head(100))

            amount_col = next(
                (c for c in ["Amount", "Transaction_Amount", "total_value"]
                 if c in df.columns),
                None
            )

            if amount_col:
                st.subheader("Fraud Amount Distribution")

                fig = px.histogram(
                    fraud_records,
                    x=amount_col,
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Dataset not found.")


# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Credit Card Fraud Model")
        show_model_info("credit_card_fraud_model.pkl")

    with col2:
        st.markdown("### Fraud Detection Model")
        show_model_info("fraud_detection_model.pkl")

    st.info(
        "Actual performance comparison requires saved evaluation metrics. "
        "This page displays the available model artifact information."
    )


# =====================================================
# FRAUD PREDICTION
# =====================================================

elif page == "Fraud Prediction":

    st.subheader("Fraud Transaction Prediction")

    model_choice = st.selectbox(
        "Choose Model",
        ["Credit Card Fraud Model", "Fraud Detection Model"]
    )

    if model_choice == "Credit Card Fraud Model":

        saved = load_model("credit_card_fraud_model.pkl")

    else:

        saved = load_model("fraud_detection_model.pkl")

    if saved is None:

        st.error("Model file not found in models folder.")

    elif not isinstance(saved, dict) or saved.get("model") is None:

        st.error("Model artifact format is not supported.")

    else:

        model = saved["model"]
        threshold = saved.get("threshold", 0.5)
        features = saved.get("features", [])

        st.write(f"Model threshold: `{threshold}`")

        if not features:
            st.error("Feature list is missing from model artifact.")

        else:

            with st.form("prediction_form"):

                values = {}

                st.write("Enter transaction details")

                for feature in features:

                    if feature in [
                        "TransactionDate",
                        "Transaction_Date"
                    ]:
                        values[feature] = st.text_input(
                            feature,
                            value=datetime.now().strftime("%Y-%m-%d")
                        )

                    elif feature in [
                        "Customer_ID",
                        "Transaction_ID",
                        "MerchantID",
                        "Merchant_Category",
                        "Payment_Method",
                        "Device_Type",
                        "Location",
                        "TransactionType",
                        "Is_International",
                        "Suspicious_Keyword"
                    ]:
                        values[feature] = st.text_input(feature)

                    else:
                        values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.4f"
                        )

                submitted = st.form_submit_button(
                    "Analyze Transaction",
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


# =====================================================
# FRAUD ANALYTICS
# =====================================================

elif page == "Fraud Analytics":

    st.subheader("Fraud Analytics")

    dataset_choice = st.selectbox(
        "Choose dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:

        target = get_target_column(df)

        if target:

            fraud_data = df[df[target] == 1]

            normal_data = df[df[target] == 0]

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud", f"{len(fraud_data):,}")
            c2.metric("Normal", f"{len(normal_data):,}")
            c3.metric(
                "Fraud Percentage",
                f"{len(fraud_data) / len(df) * 100:.2f}%"
            )

            amount_col = next(
                (c for c in ["Amount", "Transaction_Amount", "total_value"]
                 if c in df.columns),
                None
            )

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
                    fraud_data["Location"]
                    .value_counts()
                    .head(15)
                    .reset_index()
                )

                location_data.columns = ["Location", "Fraud Count"]

                fig = px.bar(
                    location_data,
                    x="Location",
                    y="Fraud Count",
                    title="Fraud Transactions by Location"
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Dataset not found.")


# =====================================================
# DETAILED INVESTIGATION
# =====================================================

elif page == "Detailed Investigation":

    st.subheader("Detailed Transaction Investigation")

    dataset_choice = st.selectbox(
        "Select dataset",
        ["Credit Card Fraud Dataset", "Fraud Detection Dataset"]
    )

    df = credit_df if dataset_choice == "Credit Card Fraud Dataset" else fraud_df

    if df is not None:

        target = get_target_column(df)

        search_col = next(
            (
                c for c in [
                    "TransactionID",
                    "Transaction_ID",
                    "Customer_ID"
                ]
                if c in df.columns
            ),
            None
        )

        amount_col = next(
            (
                c for c in [
                    "Amount",
                    "Transaction_Amount",
                    "total_value"
                ]
                if c in df.columns
            ),
            None
        )

        filtered = df.copy()

        search_value = st.text_input("Search Transaction / Customer ID")

        fraud_only = st.checkbox("Show fraud transactions only")
        high_amount = st.checkbox("Show high-value transactions (> 5000)")

        if search_value and search_col:

            filtered = filtered[
                filtered[search_col].astype(str).str.contains(
                    search_value,
                    case=False,
                    na=False
                )
            ]

        if fraud_only and target:

            filtered = filtered[filtered[target] == 1]

        if high_amount and amount_col:

            filtered = filtered[filtered[amount_col] > 5000]

        c1, c2, c3 = st.columns(3)

        c1.metric("Matching Records", f"{len(filtered):,}")

        if target:
            fraud_count = int(filtered[target].sum())
            c2.metric("Fraud Records", f"{fraud_count:,}")

        if amount_col:
            c3.metric(
                "Total Transaction Value",
                f"{filtered[amount_col].sum():,.2f}"
            )

        st.subheader("Investigation Results")
        show_dataframe(filtered.head(500))

    else:
        st.warning("Dataset not found.")


# =====================================================
# EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.subheader("Explainable AI")

    saved = load_model("credit_card_fraud_model.pkl")

    if saved is None or not isinstance(saved, dict):

        st.warning("Credit Card model artifact not found.")

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
                importance.head(20),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Model Feature Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_dataframe(importance)

        else:

            st.info(
                "Feature importance is not directly available for this model."
            )


# =====================================================
# REAL-TIME MONITORING
# =====================================================

elif page == "Real-Time Monitoring":

    st.subheader("Transaction Monitoring")

    st.info(
        "This page analyzes the available transaction dataset. "
        "It does not connect to a live transaction stream."
    )

    df = credit_df

    if df is not None:

        target = get_target_column(df)

        if target:

            fraud_count = int(df[target].sum())
            normal_count = len(df) - fraud_count

            c1, c2 = st.columns(2)

            c1.metric("Fraud Records", f"{fraud_count:,}")
            c2.metric("Normal Records", f"{normal_count:,}")

            st.subheader("Transaction Status Distribution")

            fig = go.Figure(data=[
                go.Bar(
                    x=["Normal", "Fraud"],
                    y=[normal_count, fraud_count]
                )
            ])

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Fraud target column not found.")

    else:
        st.warning("Dataset not found.")


# =====================================================
# FRAUD FORECASTING
# =====================================================

elif page == "Fraud Forecasting":

    st.subheader("Fraud Trend Analysis")

    st.info(
        "This page displays historical fraud trends, not a future prediction."
    )

    df = credit_df

    if df is not None:

        date_col = next(
            (
                c for c in [
                    "TransactionDate",
                    "Transaction_Date",
                    "date"
                ]
                if c in df.columns
            ),
            None
        )

        target = get_target_column(df)

        if date_col and target:

            trend = df.copy()

            trend[date_col] = pd.to_datetime(
                trend[date_col],
                errors="coerce"
            )

            trend = trend.dropna(subset=[date_col])

            trend = (
                trend.groupby(
                    trend[date_col].dt.date
                )[target]
                .sum()
                .reset_index()
            )

            trend.columns = ["Date", "Fraud Count"]

            fig = px.line(
                trend,
                x="Date",
                y="Fraud Count",
                markers=True,
                title="Historical Daily Fraud Transactions"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Date or fraud target column not found.")

    else:
        st.warning("Dataset not found.")


# =====================================================
# GRAPH ANALYTICS
# =====================================================

elif page == "Graph Analytics":

    st.subheader("Graph Analytics")

    st.info(
        "Basic relationship analysis using available transaction fields. "
        "This is not a dedicated graph neural network."
    )

    df = fraud_df

    if df is not None:

        relationship_options = [
            c for c in [
                "Customer_ID",
                "Merchant_Category",
                "Payment_Method",
                "Device_Type",
                "Location"
            ]
            if c in df.columns
        ]

        if relationship_options:

            selected = st.selectbox(
                "Select relationship field",
                relationship_options
            )

            counts = (
                df[selected]
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [selected, "Transaction Count"]

            fig = px.bar(
                counts,
                x=selected,
                y="Transaction Count",
                title=f"Transactions by {selected}"
            )

            st.plotly_chart(fig, use_container_width=True)

            show_dataframe(counts)

        else:
            st.warning("No supported relationship columns found.")

    else:
        st.warning("Fraud Detection dataset not found.")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:10px;">
        🛡️ FraudShield AI | Intelligent Fraud Detection Platform
        <br>
        Built with Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
