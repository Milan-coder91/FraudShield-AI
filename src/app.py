
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


def show_dataset_kpis(df):

    fraud, normal, _ = get_fraud_stats(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(df.columns))
    c3.metric("Fraud Cases", f"{fraud:,}")
    c4.metric("Normal Cases", f"{normal:,}")


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
# SIDEBAR NAVIGATION WITH ICONS
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
        <p>Real-Time Monitoring • Machine Learning • Risk Scoring
        • Fraud Investigation • Graph Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("📊 Executive Overview")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            show_fraud_chart(df)

        with col2:
            show_numeric_chart(df)

        st.divider()

        st.subheader("📁 Project Overview")

        st.write("""
        FraudShield AI is a machine-learning-based financial fraud
        detection and transaction monitoring system.

        The platform analyzes transaction data, identifies suspicious
        patterns and provides fraud probability predictions using
        trained machine learning models.
        """)

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    else:
        st.warning("Please add CSV files to the project data folders.")


# =====================================================
# 2. DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")
    st.caption("Explore dataset structure, quality and fraud distribution.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info, use_container_width=True)

        st.subheader("Dataset Quality")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        c2.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        c3.metric(
            "Memory (MB)",
            round(
                df.memory_usage(deep=True).sum() / 1024**2,
                2
            )
        )


# =====================================================
# 3. EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")
    st.caption("Fraud pattern discovery and behavioral analytics.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.divider()

        st.subheader("Fraud Distribution")
        show_fraud_chart(df)

        st.subheader("Numerical Feature Distribution")
        show_numeric_chart(df)

        st.subheader("Correlation Analysis")
        show_correlation(df)

        st.subheader("Categorical Feature Analysis")

        cat_cols = get_categorical_columns(df)

        if cat_cols:

            category = st.selectbox(
                "Select Category",
                cat_cols
            )

            top_values = (
                df[category]
                .astype(str)
                .value_counts()
                .head(15)
            )

            fig = px.bar(
                x=top_values.index,
                y=top_values.values,
                labels={
                    "x": category,
                    "y": "Count"
                },
                title=f"{category} Distribution"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No categorical columns found.")


# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering Dashboard")
    st.caption("Explore numerical and engineered variables.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        numeric_cols = get_numeric_columns(df)

        if numeric_cols:

            st.subheader("Feature Distribution")

            feature = st.selectbox(
                "Select Feature",
                numeric_cols,
                key="feature_engineering"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Mean",
                round(df[feature].mean(), 3)
            )

            col2.metric(
                "Median",
                round(df[feature].median(), 3)
            )

            col3.metric(
                "Standard Deviation",
                round(df[feature].std(), 3)
            )

            fig = px.histogram(
                df,
                x=feature,
                marginal="box",
                title=f"{feature} Analysis"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Feature Information")

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

    st.title("🚨 Fraud Intelligence")
    st.caption("Suspicious transaction patterns and risk investigation.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        fraud, normal, fraud_col = get_fraud_stats(df)

        c1, c2, c3 = st.columns(3)

        c1.metric("Fraud Cases", f"{fraud:,}")
        c2.metric("Normal Cases", f"{normal:,}")

        rate = fraud / len(df) * 100 if len(df) else 0

        c3.metric("Fraud Rate", f"{rate:.2f}%")

        st.divider()

        if fraud_col:

            st.subheader("Fraud Records")

            fraud_values = pd.to_numeric(
                df[fraud_col],
                errors="coerce"
            )

            fraud_df = df[fraud_values == 1]

            st.dataframe(
                fraud_df.head(100),
                use_container_width=True
            )

        else:
            st.info("Fraud label column not available.")

        st.subheader("High Value Transactions")

        amount_cols = [
            col for col in get_numeric_columns(df)
            if "amount" in col.lower() or "value" in col.lower()
        ]

        if amount_cols:

            amount_col = amount_cols[0]

            st.dataframe(
                df.nlargest(10, amount_col),
                use_container_width=True
            )


# =====================================================
# 6. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison")
    st.caption("Explore saved machine learning models.")

    model_files = (
        list(MODEL_DIR.glob("*.pkl"))
        if MODEL_DIR.exists()
        else []
    )

    if model_files:

        rows = []

        for file in model_files:

            try:

                saved = joblib.load(file)

                rows.append({
                    "Model": file.stem,
                    "Threshold": saved.get(
                        "threshold",
                        "Not available"
                    ),
                    "Features": len(
                        saved.get("features", [])
                    )
                })

            except Exception:

                rows.append({
                    "Model": file.stem,
                    "Threshold": "Unable to read",
                    "Features": "N/A"
                })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

        st.info(
            "This page displays saved model information. "
            "It does not calculate new evaluation metrics."
        )

    else:
        st.warning("No saved model files found.")


# =====================================================
# 7. FRAUD PREDICTION
# =====================================================

elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction")
    st.caption("Enter transaction details to predict fraud probability.")

    dataset = st.selectbox(
        "Choose Dataset",
        [
            "Credit Card Fraud Dataset",
            "Fraud Detection Dataset"
        ]
    )

    if dataset == "Credit Card Fraud Dataset":

        model_path = MODEL_DIR / "credit_card_fraud_model.pkl"

    else:

        model_path = MODEL_DIR / "fraud_detection_model.pkl"

    st.divider()
    st.header("Transaction Details")

    input_data = None

    # DATASET 1

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
            "🔍 Predict Fraud",
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
                "Is_Weekend": int(
                    transaction_date.weekday() >= 5
                )
            }

    # DATASET 2

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
            "🔍 Predict Fraud",
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
                "Is_Weekend": int(
                    transaction_date.weekday() >= 5
                )
            }

    # MODEL PREDICTION

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

                input_df = input_df.reindex(
                    columns=features
                )

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
                        st.error(
                            "⚠️ FRAUD TRANSACTION DETECTED!"
                        )
                    else:
                        st.success(
                            "✅ NORMAL TRANSACTION"
                        )

                with col2:

                    st.metric(
                        "Fraud Probability",
                        f"{fraud_probability * 100:.2f}%"
                    )

                st.metric(
                    "Model Decision Threshold",
                    f"{threshold * 100:.2f}%"
                )

                st.progress(
                    min(
                        max(float(fraud_probability), 0.0),
                        1.0
                    )
                )

                st.caption(
                    "Prediction is based on the trained model "
                    "and its validation-selected threshold. "
                    "It is not a guarantee that a transaction is safe."
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# =====================================================
# 8. FRAUD ANALYTICS
# =====================================================

elif page == "Fraud Analytics":

    st.title("📉 Fraud Analytics")
    st.caption("Transaction amount and fraud behavior analysis.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_fraud_chart(df)

        numeric_cols = get_numeric_columns(df)

        if numeric_cols:

            selected_feature = st.selectbox(
                "Select Analytics Feature",
                numeric_cols
            )

            fig = px.box(
                df,
                y=selected_feature,
                title=f"{selected_feature} Outlier Analysis"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )


# =====================================================
# 9. EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")
    st.caption("Understand model inputs and prediction behavior.")

    model_files = (
        list(MODEL_DIR.glob("*.pkl"))
        if MODEL_DIR.exists()
        else []
    )

    if model_files:

        selected_model = st.selectbox(
            "Select Model",
            model_files,
            format_func=lambda x: x.name
        )

        try:

            saved = joblib.load(selected_model)

            features = saved.get("features", [])
            threshold = saved.get("threshold", None)

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.2f}%"
                if threshold is not None
                else "N/A"
            )

            st.subheader("Model Input Features")

            st.dataframe(
                pd.DataFrame({
                    "Feature": features
                }),
                use_container_width=True
            )

            model = saved.get("model")

            if hasattr(model, "feature_importances_"):

                importance = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(
                    "Importance",
                    ascending=False
                )

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Feature importance is not directly available "
                    "for this saved model pipeline."
                )

        except Exception as e:

            st.error(f"Model information error: {e}")

    else:

        st.warning("No saved models found.")


# =====================================================
# 10. REAL-TIME MONITORING
# =====================================================

elif page == "Real-Time Monitoring":

    st.title("📡 Real-Time Monitoring")
    st.caption("Monitor available transaction records.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        show_dataset_kpis(df)

        st.subheader("Latest Available Records")

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

        st.info(
            "This page displays records currently stored in CSV files. "
            "Live transaction streaming is not connected."
        )


# =====================================================
# 11. FRAUD FORECASTING
# =====================================================

elif page == "Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")
    st.caption("Explore transaction trends over time.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        date_cols = [
            col for col in df.columns
            if "date" in col.lower() or "time" in col.lower()
        ]

        if date_cols:

            date_col = st.selectbox(
                "Select Date Column",
                date_cols
            )

            temp = df.copy()

            temp[date_col] = pd.to_datetime(
                temp[date_col],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[date_col]
            )

            if not temp.empty:

                temp["Date"] = temp[date_col].dt.date

                daily = (
                    temp.groupby("Date")
                    .size()
                    .reset_index(name="Transactions")
                )

                fig = px.line(
                    daily,
                    x="Date",
                    y="Transactions",
                    title="Daily Transaction Trend",
                    markers=True
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1120"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info("No valid date records found.")

        else:

            st.info("No date or time column found in this dataset.")

        st.warning(
            "This page shows historical trends only. "
            "A forecasting model has not been connected here."
        )


# =====================================================
# 12. GRAPH ANALYTICS
# =====================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics")
    st.caption("Explore relationships between transaction entities.")

    selected = show_dataset_selector()

    if selected:

        df = datasets[selected]

        entity_cols = [
            col for col in df.columns
            if any(
                key in col.lower()
                for key in [
                    "customer",
                    "merchant",
                    "device",
                    "location",
                    "account"
                ]
            )
        ]

        if entity_cols:

            st.subheader("Available Entity Columns")

            st.write(entity_cols)

            selected_entity = st.selectbox(
                "Select Entity",
                entity_cols
            )

            entity_counts = (
                df[selected_entity]
                .astype(str)
                .value_counts()
                .head(15)
                .reset_index()
            )

            entity_counts.columns = [
                "Entity",
                "Transactions"
            ]

            fig = px.bar(
                entity_counts,
                x="Entity",
                y="Transactions",
                title="Top Entities by Transaction Count"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No customer, merchant or entity columns detected."
            )

        st.caption(
            "This is entity frequency analysis, not a connected graph network."
        )


# =====================================================
# FOOTER
# =====================================================

st.sidebar.divider()
st.sidebar.caption("🛡️ FraudShield AI")
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("© 2026 FraudShield AI")
