import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from pathlib import Path
from datetime import date

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

h1, h2, h3 {
    color: #f8fafc;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172554, #1e293b);
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetric"] label {
    color: #93c5fd;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.hero {
    background: linear-gradient(110deg, #123b91, #1766d1);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 22px;
    color: white;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

/* Remove red focus outline/glow after clicking */
div.stButton > button:focus,
div.stButton > button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border-color: #60a5fa !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.prediction-card {
    padding: 18px 20px;
    border: 1px solid #3b82f6;
    border-left: 5px solid #60a5fa;
    border-radius: 12px;
    background: linear-gradient(110deg, #172554, #1e293b);
    color: #f8fafc;
    margin: 8px 0 14px 0;
}

.prediction-card h3 {
    margin: 0 0 6px 0;
    color: #ffffff;
}

.prediction-card p {
    margin: 0;
    color: #dbeafe;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASETS
# =====================================================

@st.cache_data
def load_datasets():

    datasets = {}

    for folder in [RAW_DIR, PROCESSED_DIR]:

        if folder.exists():

            for file in folder.glob("*.csv"):

                try:
                    df = pd.read_csv(file, low_memory=False)

                    if not df.empty:
                        datasets[file.stem] = df

                except Exception:
                    pass

    return datasets


datasets = load_datasets()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def find_fraud_column(df):

    possible = [
        "IsFraud",
        "Fraudulent",
        "is_fraud",
        "fraud",
        "Fraud",
        "isFraud"
    ]

    for col in possible:
        if col in df.columns:
            return col

    return None


def get_fraud_stats(df):

    fraud_col = find_fraud_column(df)

    if fraud_col is None:
        return 0, 0, None

    values = df[fraud_col]

    if values.dtype == "object":
        values = values.astype(str).str.lower().map({
            "1": 1,
            "true": 1,
            "yes": 1,
            "fraud": 1,
            "0": 0,
            "false": 0,
            "no": 0,
            "normal": 0
        })

    values = pd.to_numeric(values, errors="coerce")

    fraud = int((values == 1).sum())
    normal = int((values == 0).sum())

    return fraud, normal, fraud_col


def get_numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


def show_dataset_selector():

    if not datasets:
        st.warning(
            "No CSV datasets found in data/raw or data/processed."
        )
        return None

    return st.selectbox(
        "Select Dataset",
        list(datasets.keys())
    )


# =====================================================
# ADVANCED DATASET KPIs
# =====================================================

def show_dataset_kpis(df):

    fraud, normal, fraud_col = get_fraud_stats(df)

    amount_cols = [
        col for col in get_numeric_columns(df)
        if any(
            key in col.lower()
            for key in ["amount", "total_value", "value"]
        )
    ]

    amount_col = amount_cols[0] if amount_cols else None

    fraud_rate = (
        fraud / len(df) * 100
        if len(df) and fraud_col else 0
    )

    avg_amount = (
        pd.to_numeric(
            df[amount_col],
            errors="coerce"
        ).mean()
        if amount_col else None
    )

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    kpis = [
        ("Total Transactions", f"{len(df):,}"),
        ("Total Features", f"{len(df.columns):,}"),
        ("Fraud Cases", f"{fraud:,}" if fraud_col else "N/A"),
        ("Normal Cases", f"{normal:,}" if fraud_col else "N/A"),
        ("Fraud Rate", f"{fraud_rate:.2f}%" if fraud_col else "N/A"),
        (
            "Average Transaction Value",
            f"{avg_amount:,.2f}"
            if pd.notna(avg_amount) else "N/A"
        )
    ]

    for row_start in range(0, len(kpis), 3):

        cols = st.columns(3)

        for col, (label, value) in zip(
            cols,
            kpis[row_start:row_start + 3]
        ):

            with col:
                st.metric(label, value)

    st.caption(
        f"Dataset Quality: {missing:,} missing values "
        f"• {duplicates:,} duplicate rows"
    )


# =====================================================
# FRAUD DISTRIBUTION CHART
# =====================================================

def show_fraud_chart(df):

    fraud, normal, fraud_col = get_fraud_stats(df)

    if fraud_col is None:
        st.info("Fraud label column not found.")
        return

    chart_df = pd.DataFrame({
        "Transaction Type": ["Fraud", "Normal"],
        "Count": [fraud, normal]
    })

    fig = px.pie(
        chart_df,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        title="Fraud vs Normal Distribution",
        color="Transaction Type",
        color_discrete_map={
            "Fraud": "#ef4444",
            "Normal": "#2563eb"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# NUMERICAL FEATURE CHART
# =====================================================

def show_numeric_chart(df):

    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        st.info("No numerical columns available.")
        return

    selected = st.selectbox(
        "Select Numerical Feature",
        numeric_cols,
        key="numeric_chart"
    )

    fig = px.histogram(
        df,
        x=selected,
        title=f"{selected} Distribution",
        color_discrete_sequence=["#3b82f6"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def show_correlation(df):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.info("At least two numerical columns are required.")
        return

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("## 🛡️ FraudShield AI")
st.sidebar.caption("Enterprise Fraud Detection Platform")

pages = {
    "📊 Executive Dashboard": "Executive Dashboard",
    "📁 Dataset Overview": "Dataset Overview",
    "📈 EDA Dashboard": "EDA Dashboard",
    "⚙️ Feature Engineering": "Feature Engineering",
    "🚨 Fraud Intelligence": "Fraud Intelligence",
    "🤖 Model Comparison": "Model Comparison",
    "🔍 Fraud Prediction": "Fraud Prediction",
    "📉 Fraud Analytics": "Fraud Analytics",
    "🧠 Explainable AI": "Explainable AI",
    "📡 Real-Time Monitoring": "Real-Time Monitoring",
    "🔮 Fraud Forecasting": "Fraud Forecasting",
    "🕸️ Graph Analytics": "Graph Analytics"
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(pages.keys())
)

page = pages[selected_page]

st.sidebar.divider()
st.sidebar.markdown("### 🛡️ System Status")
st.sidebar.success("Dashboard Online")
st.sidebar.caption("Machine Learning Fraud Detection")
# =====================================================
# 1. EXECUTIVE DASHBOARD
# =====================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ FraudShield AI</h1>
        <h3>Enterprise Fraud Detection & Risk Intelligence Platform</h3>
        <p>
            Machine Learning • Fraud Detection • Risk Scoring
            • Transaction Analytics • Fraud Investigation
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.header("📊 Executive Overview")

    st.caption(
        "A consolidated view of transaction volume, fraud exposure "
        "and dataset quality."
    )

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        fraud, normal, fraud_col = get_fraud_stats(df)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            show_fraud_chart(df)

        with col2:
            show_numeric_chart(df)

        st.divider()

        st.subheader("📈 Fraud Risk Summary")

        if fraud_col:

            fraud_rate = (
                fraud / len(df) * 100
                if len(df) else 0
            )

            normal_rate = (
                normal / len(df) * 100
                if len(df) else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Fraud Percentage",
                f"{fraud_rate:.2f}%"
            )

            c2.metric(
                "Normal Percentage",
                f"{normal_rate:.2f}%"
            )

            c3.metric(
                "Fraud to Normal Ratio",
                f"{fraud}:{normal}"
            )

        st.divider()

        st.subheader("📁 Project Overview")

        st.markdown("""
        FraudShield AI is a machine-learning-based financial fraud
        detection and transaction monitoring platform.

        The system analyzes transaction data, identifies suspicious
        patterns and estimates fraud probability using trained
        Random Forest models.

        The platform supports:
        - Transaction-level fraud prediction
        - Fraud pattern analysis
        - Dataset quality monitoring
        - Feature importance analysis
        - Model threshold inspection
        """)

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    else:

        st.warning(
            "Please add CSV files to the project data folders."
        )


# =====================================================
# 2. DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")

    st.caption(
        "Explore dataset structure, quality and fraud distribution."
    )

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.divider()

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.divider()

        st.subheader("🔍 Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(
            info,
            use_container_width=True
        )

        st.divider()

        st.subheader("🧹 Dataset Quality Analysis")

        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        total_cells = df.shape[0] * df.shape[1]

        missing_percentage = (
            missing / total_cells * 100
            if total_cells else 0
        )

        unique_columns = int(
            df.nunique().sum()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Missing Values",
            f"{missing:,}"
        )

        c2.metric(
            "Duplicate Rows",
            f"{duplicates:,}"
        )

        c3.metric(
            "Missing Data %",
            f"{missing_percentage:.2f}%"
        )

        st.metric(
            "Total Unique Values Across Features",
            f"{unique_columns:,}"
        )

        if missing == 0 and duplicates == 0:

            st.success(
                "Dataset quality check: No missing values "
                "or duplicate rows detected."
            )

        else:

            st.warning(
                "Dataset contains missing values or duplicate rows."
            )


# =====================================================
# 3. EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    st.caption(
        "Fraud pattern discovery and behavioral analytics."
    )

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.divider()

        st.subheader("🚨 Fraud Distribution")

        show_fraud_chart(df)

        st.divider()

        st.subheader("📊 Numerical Feature Distribution")

        show_numeric_chart(df)

        st.divider()

        st.subheader("🔥 Correlation Analysis")

        show_correlation(df)

        st.divider()

        st.subheader("📂 Categorical Feature Analysis")

        cat_cols = get_categorical_columns(df)

        if cat_cols:

            category = st.selectbox(
                "Select Categorical Feature",
                cat_cols,
                key="eda_category"
            )

            top_values = (
                df[category]
                .astype(str)
                .value_counts()
                .head(15)
            )

            chart_df = top_values.reset_index()

            chart_df.columns = [
                "Category",
                "Count"
            ]

            fig = px.bar(
                chart_df,
                x="Category",
                y="Count",
                title=f"{category} Distribution",
                color="Count",
                color_continuous_scale="Blues"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No categorical columns found."
            )


# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering Dashboard")

    st.caption(
        "Explore numerical variables, feature statistics "
        "and dataset characteristics."
    )

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        numeric_cols = get_numeric_columns(df)

        if numeric_cols:

            st.divider()

            st.subheader("📊 Feature Analysis")

            feature = st.selectbox(
                "Select Feature",
                numeric_cols,
                key="feature_engineering"
            )

            values = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).dropna()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Mean",
                f"{values.mean():,.3f}"
                if not values.empty else "N/A"
            )

            c2.metric(
                "Median",
                f"{values.median():,.3f}"
                if not values.empty else "N/A"
            )

            c3.metric(
                "Standard Deviation",
                f"{values.std():,.3f}"
                if not values.empty else "N/A"
            )

            st.divider()

            if not values.empty:

                fig = px.histogram(
                    values,
                    x=feature,
                    marginal="box",
                    title=f"{feature} Distribution",
                    color_discrete_sequence=["#3b82f6"]
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#0b1120"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("📋 Feature Statistics")

            stats = df[numeric_cols].describe().T

            st.dataframe(
                stats,
                use_container_width=True
            )

        else:

            st.info(
                "No numerical features available."
            )

        st.divider()

        st.subheader("🧾 Feature Information")

        st.dataframe(
            pd.DataFrame({
                "Feature": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values
            }),
            use_container_width=True
        )


# =====================================================
# 5. FRAUD INTELLIGENCE
# =====================================================

elif page == "Fraud Intelligence":

    st.title("🚨 Fraud Intelligence Center")

    st.caption(
        "Suspicious transaction patterns and risk investigation."
    )

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        fraud, normal, fraud_col = get_fraud_stats(df)

        fraud_rate = (
            fraud / len(df) * 100
            if len(df) and fraud_col else 0
        )

        st.subheader("🛡️ Fraud Intelligence KPIs")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fraud Cases",
            f"{fraud:,}" if fraud_col else "N/A"
        )

        c2.metric(
            "Normal Cases",
            f"{normal:,}" if fraud_col else "N/A"
        )

        c3.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%" if fraud_col else "N/A"
        )

        st.divider()

        if fraud_col:

            st.subheader("📊 Fraud vs Normal Analysis")

            show_fraud_chart(df)

            fraud_values = pd.to_numeric(
                df[fraud_col],
                errors="coerce"
            )

            fraud_df = df[fraud_values == 1]

            st.subheader("🔍 Fraud Records")

            st.caption(
                f"Showing up to 100 fraud records from "
                f"{len(fraud_df):,} detected fraud-labelled rows."
            )

            st.dataframe(
                fraud_df.head(100),
                use_container_width=True
            )

        else:

            st.info(
                "Fraud label column is not available in this dataset."
            )

        st.divider()

        st.subheader("💰 High Value Transactions")

        amount_cols = [
            col for col in get_numeric_columns(df)
            if (
                "amount" in col.lower()
                or "value" in col.lower()
            )
        ]

        if amount_cols:

            amount_col = st.selectbox(
                "Select Transaction Value Feature",
                amount_cols,
                key="intelligence_amount"
            )

            temp = df.copy()

            temp[amount_col] = pd.to_numeric(
                temp[amount_col],
                errors="coerce"
            )

            high_value_df = (
                temp.nlargest(10, amount_col)
            )

            st.dataframe(
                high_value_df,
                use_container_width=True
            )

        else:

            st.info(
                "No transaction amount or value column found."
            )




elif page == "Model Comparison":

    st.title("⚖️ Model Comparison")
    st.caption("Saved model artifacts and configuration")

    model_files = {
        "Credit Card Fraud Model": MODEL_DIR / "credit_card_fraud_model.pkl",
        "Fraud Detection Model": MODEL_DIR / "fraud_detection_model.pkl",
    }

    cols = st.columns(2)

    for i, (model_name, model_path) in enumerate(model_files.items()):
        with cols[i]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### {model_name}")

            if model_path.exists():
                st.markdown(
                    '<span style="color:#22c55e;">● Model Available</span>',
                    unsafe_allow_html=True
                )

                try:
                    saved_data = joblib.load(model_path)
                    model = saved_data.get("model")
                    threshold = saved_data.get("threshold")
                    features = saved_data.get("features", [])

                    st.write(f"**Features:** {len(features)}")

                    if threshold is not None:
                        st.write(f"**Decision Threshold:** {threshold:.4f}")

                    st.write(
                        f"**Model Type:** {type(model).__name__}"
                        if model is not None else "**Model Type:** Not available"
                    )

                except Exception as e:
                    st.warning(f"Model information could not be loaded: {e}")

            else:
                st.warning("Model file not found.")

            st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "This page displays saved model information. "
        "It does not calculate new evaluation metrics."
    )


elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction")
    st.caption("Check a transaction using your trained Random Forest models.")

    dataset_choice = st.selectbox(
        "Select Prediction Model",
        [
            "Credit Card Fraud Dataset",
            "Fraud Detection Dataset"
        ]
    )

    st.divider()

    if dataset_choice == "Credit Card Fraud Dataset":

        st.subheader("Credit Card Transaction Details")

        with st.form("credit_card_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                transaction_id = st.text_input("Transaction ID")
                amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=1000.0
                )
                merchant_id = st.number_input(
                    "Merchant ID",
                    min_value=0,
                    value=100
                )
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["Online", "POS", "ATM"]
                )

            with col2:
                location = st.selectbox(
                    "Location",
                    ["India", "USA", "UK", "Canada", "Other"]
                )
                transaction_date = st.date_input(
                    "Transaction Date"
                )

            submitted = st.form_submit_button(
                "🔍 Analyze Transaction",
                use_container_width=True
            )

        if submitted:

            model_path = MODEL_DIR / "credit_card_fraud_model.pkl"

            if not model_path.exists():
                st.warning("Credit card model file not found.")

            else:
                try:
                    saved_data = joblib.load(model_path)

                    model = saved_data["model"]
                    threshold = saved_data["threshold"]
                    features = saved_data["features"]

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

                    input_df = pd.DataFrame([input_data]).reindex(
                        columns=features
                    )

                    probability = model.predict_proba(input_df)[0][1]
                    prediction = int(probability >= threshold)

                    st.markdown(
                        f"""
                        <div style="
                            background:#172554;
                            border:1px solid #3b82f6;
                            border-radius:12px;
                            padding:22px;
                            margin-top:20px;
                        ">
                            <h3 style="color:#ffffff;">
                                Prediction Result
                            </h3>
                            <h2 style="color:#ffffff;">
                                {"Fraudulent Transaction" if prediction == 1 else "Normal Transaction"}
                            </h2>
                            <p style="color:#dbeafe;">
                                Fraud Probability: {probability * 100:.2f}%
                            </p>
                            <p style="color:#dbeafe;">
                                Decision Threshold: {threshold:.4f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.warning(f"Prediction could not be completed: {e}")

    else:

        st.subheader("Fraud Detection Transaction Details")

        with st.form("fraud_detection_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                customer_id = st.number_input(
                    "Customer ID",
                    min_value=0,
                    value=1001
                )
                transaction_amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=1500.0
                )
                merchant_category = st.selectbox(
                    "Merchant Category",
                    ["Retail", "Food", "Travel", "Online", "Other"]
                )
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Credit Card", "Debit Card", "UPI", "Cash", "Other"]
                )
                device_type = st.selectbox(
                    "Device Type",
                    ["Mobile", "Desktop", "Tablet", "Other"]
                )
                location = st.selectbox(
                    "Location",
                    ["India", "USA", "UK", "Canada", "Other"]
                )

            with col2:
                is_international = st.selectbox(
                    "International Transaction",
                    ["No", "Yes"]
                )
                previous_transactions = st.number_input(
                    "Previous Transactions",
                    min_value=0,
                    value=10
                )
                average_spend = st.number_input(
                    "Average Spend",
                    min_value=0.0,
                    value=2000.0
                )
                account_age_days = st.number_input(
                    "Account Age (Days)",
                    min_value=0,
                    value=365
                )
                suspicious_keyword = st.selectbox(
                    "Suspicious Keyword",
                    ["No", "Yes"]
                )
                transaction_date = st.date_input(
                    "Transaction Date"
                )

            submitted = st.form_submit_button(
                "🔍 Analyze Transaction",
                use_container_width=True
            )

        if submitted:

            model_path = MODEL_DIR / "fraud_detection_model.pkl"

            if not model_path.exists():
                st.warning("Fraud detection model file not found.")

            else:
                try:
                    saved_data = joblib.load(model_path)

                    model = saved_data["model"]
                    threshold = saved_data["threshold"]
                    features = saved_data["features"]

                    input_data = {
                        "Customer_ID": customer_id,
                        "Transaction_Amount": transaction_amount,
                        "Merchant_Category": merchant_category,
                        "Payment_Method": payment_method,
                        "Device_Type": device_type,
                        "Location": location,
                        "Is_International": int(is_international == "Yes"),
                        "Previous_Transactions": previous_transactions,
                        "Average_Spend": average_spend,
                        "Account_Age_Days": account_age_days,
                        "Suspicious_Keyword": int(suspicious_keyword == "Yes"),
                        "Transaction_Month": transaction_date.month,
                        "Transaction_Day": transaction_date.day,
                        "Transaction_Hour": 0,
                        "Transaction_Weekday": transaction_date.weekday(),
                        "Is_Weekend": int(transaction_date.weekday() >= 5)
                    }

                    input_df = pd.DataFrame([input_data]).reindex(
                        columns=features
                    )

                    probability = model.predict_proba(input_df)[0][1]
                    prediction = int(probability >= threshold)

                    st.markdown(
                        f"""
                        <div style="
                            background:#172554;
                            border:1px solid #3b82f6;
                            border-radius:12px;
                            padding:22px;
                            margin-top:20px;
                        ">
                            <h3 style="color:#ffffff;">
                                Prediction Result
                            </h3>
                            <h2 style="color:#ffffff;">
                                {"Fraudulent Transaction" if prediction == 1 else "Normal Transaction"}
                            </h2>
                            <p style="color:#dbeafe;">
                                Fraud Probability: {probability * 100:.2f}%
                            </p>
                            <p style="color:#dbeafe;">
                                Decision Threshold: {threshold:.4f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.warning(f"Prediction could not be completed: {e}")


elif page == "Fraud Analytics":

    st.title("📊 Fraud Analytics")

    if df is None or df.empty:
        st.warning("Please select a dataset containing transaction records.")

    else:
        fraud_col = next(
            (
                col for col in
                ["IsFraud", "Fraudulent", "is_fraud", "fraud"]
                if col in df.columns
            ),
            None
        )

        amount_col = next(
            (
                col for col in
                ["Amount", "Transaction_Amount", "total_value"]
                if col in df.columns
            ),
            None
        )

        if fraud_col:

            fraud_data = df.copy()
            fraud_data[fraud_col] = pd.to_numeric(
                fraud_data[fraud_col],
                errors="coerce"
            )

            fraud_counts = fraud_data[fraud_col].value_counts().sort_index()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fraud Cases",
                    f"{int(fraud_data[fraud_col].eq(1).sum()):,}"
                )

            with col2:
                fraud_rate = fraud_data[fraud_col].eq(1).mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.subheader("Fraud vs Normal Transactions")

            chart_data = pd.DataFrame({
                "Transaction Type": [
                    "Normal" if x == 0 else "Fraud"
                    for x in fraud_counts.index
                ],
                "Count": fraud_counts.values
            })

            st.bar_chart(
                chart_data.set_index("Transaction Type")
            )

            if amount_col:

                st.subheader("Transaction Amount Analysis")

                amount_data = fraud_data[
                    [fraud_col, amount_col]
                ].copy()

                amount_data[amount_col] = pd.to_numeric(
                    amount_data[amount_col],
                    errors="coerce"
                )

                amount_data = amount_data.dropna()

                if not amount_data.empty:
                    amount_data["Transaction Type"] = amount_data[
                        fraud_col
                    ].map({
                        0: "Normal",
                        1: "Fraud"
                    })

                    st.bar_chart(
                        amount_data.groupby(
                            "Transaction Type"
                        )[amount_col].mean()
                    )

                    st.write(
                        "Average transaction amount by transaction type."
                    )

        else:
            st.info("Fraud label column was not found in this dataset.")


elif page == "Explainable AI":

    st.title("🧠 Explainable AI")
    st.caption("Explore feature importance from available trained models.")

    explain_model = st.selectbox(
        "Select Model",
        [
            "Credit Card Fraud Model",
            "Fraud Detection Model"
        ]
    )

    if explain_model == "Credit Card Fraud Model":
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
                estimator = list(model.named_steps.values())[-1]

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

                    st.bar_chart(
                        importance_df.set_index("Feature").head(15)
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.info(
                        "Feature names and importance values do not match."
                    )

            else:
                st.info(
                    "Feature importance is not directly available "
                    "for this saved model."
                )

        except Exception as e:
            st.warning(f"Could not inspect the model: {e}")

    else:
        st.warning("Selected model file was not found.")


elif page == "Real-Time Monitoring":

    st.title("📡 Transaction Monitoring")

    if df is None or df.empty:
        st.warning("No transaction data available.")

    else:
        st.subheader("Latest Transaction Records")

        date_col = next(
            (
                col for col in
                ["TransactionDate", "Transaction_Date", "date"]
                if col in df.columns
            ),
            None
        )

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
            monitoring_df.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This page monitors records from the selected CSV dataset. "
            "It is not connected to a live transaction stream."
        )


elif page == "Fraud Forecasting":

    st.title("📈 Fraud Trend Analysis")

    if df is None or df.empty:
        st.warning("No transaction data available.")

    else:
        date_col = next(
            (
                col for col in
                ["TransactionDate", "Transaction_Date", "date"]
                if col in df.columns
            ),
            None
        )

        fraud_col = next(
            (
                col for col in
                ["IsFraud", "Fraudulent", "is_fraud", "fraud"]
                if col in df.columns
            ),
            None
        )

        if date_col and fraud_col:

            trend_df = df[[date_col, fraud_col]].copy()

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

                trend_df["Date"] = trend_df[date_col].dt.date

                daily_fraud = trend_df.groupby("Date")[
                    fraud_col
                ].sum()

                st.subheader("Daily Fraud Transactions")

                st.line_chart(daily_fraud)

                st.subheader("Daily Transaction Volume")

                daily_total = trend_df.groupby("Date").size()

                st.line_chart(daily_total)

            else:
                st.info("No valid date and fraud records found.")

        else:
            st.info(
                "Date column or fraud label column is missing."
            )

        st.info(
            "This page displays historical trends. "
            "It does not generate future predictions."
        )


elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")

    if df is None or df.empty:
        st.warning("No dataset selected.")

    else:
        st.subheader("Transaction Entity Analysis")

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

            entity_counts = df[selected_entity].value_counts().head(15)

            st.bar_chart(entity_counts)

            st.dataframe(
                entity_counts.rename("Transaction Count"),
                use_container_width=True
            )

        else:
            st.info(
                "No supported customer, merchant, or location columns found."
            )

        st.info(
            "This page shows entity frequency analysis. "
            "It does not calculate actual network connections between entities."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding:15px;
    ">
        <p><b>FraudShield AI</b> | AI-Powered Fraud Detection Dashboard</p>
        <p>Built with Python, Scikit-learn and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
