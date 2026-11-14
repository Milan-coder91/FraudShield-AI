import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import re
from pathlib import Path
from datetime import datetime

# =========================================================
# 1. CONFIGURATION
# =========================================================

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

OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_FILES = {
    "Credit Card Fraud Dataset": "credit_card_fraud_dataset.csv",
    "Fraud Detection Dataset": "fraud_detection_dataset.csv"
}

MODEL_FILES = {
    "Credit Card Fraud": "credit_card_fraud_model.pkl",
    "Credit Card Fraud SMOTE": "credit_card_fraud_smote_model.pkl",
    "Fraud Detection": "fraud_detection_model.pkl",
    "Fraud Detection SMOTE": "fraud_detection_smote_model.pkl"
}

# =========================================================
# 2. STYLING
# =========================================================

st.markdown("""
<style>
.stApp {
    background: #080f1b;
    color: #eaf1fa;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020817,#10223a);
    border-right: 1px solid #263c56;
}
[data-testid="stSidebar"] * {
    color: #e4efff;
}
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 5px;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px;
    border-radius: 8px;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1a3554;
}
.hero {
    background: linear-gradient(115deg,#0799e8,#1761e8 58%,#174bd0);
    padding: 48px 42px;
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(0,80,200,.23);
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
    background: linear-gradient(145deg,#111e30,#0d1929);
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
.pipeline-container,.feature-container {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    overflow-x: auto;
    padding: 14px 2px;
}
.pipeline-step,.feature-item {
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
    color: white;
}
.footer {
    text-align: center;
    color: #8295ac;
    padding: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA FUNCTIONS
# =========================================================

@st.cache_data(show_spinner=False)
def read_csv_file(file_bytes):
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return None


def find_dataset(filename):
    search_folders = [
        RAW_DIR,
        PROCESSED_DIR,
        DATA_DIR,
        BASE_DIR
    ]

    for folder in search_folders:
        candidate = folder / filename
        if candidate.exists():
            return candidate

    for folder in search_folders:
        if folder.exists():
            matches = list(folder.rglob(filename))
            if matches:
                return matches[0]

    return None


def load_default_datasets():
    loaded = {}

    for display_name, filename in DATASET_FILES.items():
        path = find_dataset(filename)

        if path:
            try:
                loaded[display_name] = pd.read_csv(path)
            except Exception as error:
                st.warning(f"Could not load {filename}: {error}")

    return loaded


if "uploaded_datasets" not in st.session_state:
    st.session_state.uploaded_datasets = {}

datasets = load_default_datasets()
datasets.update(st.session_state.uploaded_datasets)


def target_column(df):
    possible = [
        "IsFraud", "Fraudulent", "is_fraud",
        "fraud", "Class"
    ]
    return next((c for c in possible if c in df.columns), None)


def amount_column(df):
    possible = [
        "Amount", "Transaction_Amount",
        "total_value", "TransactionAmount"
    ]
    return next((c for c in possible if c in df.columns), None)


def date_column(df):
    possible = [
        "TransactionDate", "Transaction_Date",
        "date", "Transaction_Date_Time"
    ]
    return next((c for c in possible if c in df.columns), None)


def fraud_mask(df):
    target = target_column(df)

    if target is None:
        return pd.Series(False, index=df.index)

    values = pd.to_numeric(df[target], errors="coerce").fillna(0)
    return values == 1


def fraud_count(df):
    return int(fraud_mask(df).sum())


def normal_count(df):
    return max(0, len(df) - fraud_count(df))


def fraud_rate(df):
    return fraud_count(df) / len(df) * 100 if len(df) else 0


def model_files():
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*.pkl"))


def load_model_file(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


def get_model(saved):
    if isinstance(saved, dict):
        return saved.get("model")
    return saved


def get_features(saved, model):
    if isinstance(saved, dict) and saved.get("features") is not None:
        return list(saved["features"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return []


def show_chart(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#080f1b",
        plot_bgcolor="#101d2d",
        font=dict(color="#eaf1fa", family="Arial"),
        title=dict(font=dict(size=20, color="white"), x=0.02),
        margin=dict(l=35, r=25, t=65, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#26394f"),
        yaxis=dict(gridcolor="#26394f")
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)


def show_table(df, rows=500):
    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


def dataset_selector(key="dataset_selector"):
    if not datasets:
        return None

    return st.selectbox(
        "Select Dataset",
        list(datasets.keys()),
        key=key
    )


# =========================================================
# 4. PDF REPORT GENERATOR
# =========================================================

def create_pdf_report(df, title, dataset_name):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )

        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=35,
            leftMargin=35,
            topMargin=35,
            bottomMargin=35
        )

        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("FraudShield AI", styles["Title"]))
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(
            f"Dataset: {dataset_name}", styles["Normal"]
        ))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        summary = [
            ["Metric", "Value"],
            ["Total Transactions", f"{len(df):,}"],
            ["Fraud Cases", f"{fraud_count(df):,}"],
            ["Normal Transactions", f"{normal_count(df):,}"],
            ["Fraud Rate", f"{fraud_rate(df):.2f}%"],
            ["Columns", str(df.shape[1])],
            ["Missing Values", str(int(df.isnull().sum().sum()))],
            ["Duplicate Rows", str(int(df.duplicated().sum()))]
        ]

        table = Table(summary, colWidths=[230, 220])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174bd0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.lightgrey])
        ]))

        story.append(table)
        story.append(Spacer(1, 18))

        story.append(Paragraph("Column Information", styles["Heading2"]))

        column_data = [["Column", "Data Type", "Missing", "Unique"]]

        for col in df.columns[:35]:
            column_data.append([
                str(col)[:35],
                str(df[col].dtype),
                str(int(df[col].isnull().sum())),
                str(int(df[col].nunique()))
            ])

        column_table = Table(
            column_data,
            repeatRows=1,
            colWidths=[170, 90, 75, 75]
        )

        column_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174bd0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))

        story.append(column_table)
        story.append(Spacer(1, 15))
        story.append(Paragraph(
            "Report generated by FraudShield AI.",
            styles["Italic"]
        ))

        document.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        st.error("PDF library missing. Add reportlab to requirements.txt.")
        return None

    except Exception as error:
        st.error(f"PDF generation failed: {error}")
        return None


def report_download_button(df, dataset_name, report_title="Dataset Report"):
    pdf = create_pdf_report(df, report_title, dataset_name)

    if pdf:
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", f"{dataset_name}_{report_title}").strip("_")
        st.download_button(
            "📄 Download PDF Report",
            data=pdf,
            file_name="FraudShield_AI_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_download_{safe_name}"
        )


# =========================================================
# 5. PAGE ROUTING
# =========================================================
# The selected page is injected by the root app.py or a pages/*.py wrapper.
if "page" not in globals():
    page = "🏠 app"


# =========================================================
# 6. HOME PAGE
# =========================================================

if page == "🏠 app":

    # HERO BANNER

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

    # PROJECT OVERVIEW

    st.markdown("""
    <div class="project-card">
        <div class="project-heading">🚀 Project Overview</div>
        <div class="project-text">
            FraudShield AI is a financial fraud detection platform designed to
            identify suspicious transactions using Machine Learning and data analytics.
            It provides dataset analysis, fraud prediction, transaction investigation,
            model insights and reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI FRAUD DETECTION PIPELINE

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

    # PLATFORM FEATURES

    st.subheader("✨ Platform Features")

    features = [
        "📊 Executive Dashboard",
        "📁 Dataset Analysis",
        "📈 EDA & Charts",
        "🤖 ML Prediction",
        "🔎 Investigation",
        "🧠 Explainable AI",
        "📡 Monitoring",
        "📄 PDF Reports",
        "📤 Data Upload",
        "🔗 Graph Analytics"
    ]

    feature_cols = st.columns(5)

    for i, feature in enumerate(features):
        with feature_cols[i % 5]:
            st.info(feature)

    st.divider()

    # DASHBOARD FILTERS

    st.subheader("📅 Dashboard Filters")

    selected_home = dataset_selector(key="home_dataset_selector")

    if selected_home:
        source_df = datasets[selected_home].copy()
        home_df = source_df.copy()
        date_col = date_column(home_df)
        target_col = target_column(home_df)
        amount_col = amount_column(home_df)

        if date_col:
            home_df[date_col] = pd.to_datetime(home_df[date_col], errors="coerce")
            valid_dates = home_df[date_col].dropna()

            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1.2])

                with f1:
                    date_range = st.date_input(
                        "Start Date – End Date",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="home_date_range"
                    )

                with f2:
                    month_names = [
                        "January", "February", "March", "April",
                        "May", "June", "July", "August",
                        "September", "October", "November", "December"
                    ]

                    selected_months = st.multiselect(
                        "Months",
                        month_names,
                        default=month_names,
                        key="home_month_filter"
                    )

                with f3:
                    available_years = sorted(
                        valid_dates.dt.year.unique().tolist()
                    )

                    year_options = ["All Years"] + [
                        str(y) for y in available_years
                    ]

                    selected_year = st.selectbox(
                        "Year",
                        year_options,
                        key="home_year_filter"
                    )

                with f4:
                    st.caption(
                        f"Available data: {min_date} to {max_date}"
                    )

                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                    start_date, end_date = date_range

                    home_df = home_df[
                        home_df[date_col].dt.date.between(
                            start_date, end_date
                        )
                    ]

                month_numbers = [
                    month_names.index(month) + 1
                    for month in selected_months
                ]

                home_df = home_df[
                    home_df[date_col].dt.month.isin(month_numbers)
                ]

                if selected_year != "All Years":
                    home_df = home_df[
                        home_df[date_col].dt.year == int(selected_year)
                    ]

            else:
                st.warning(
                    "Date column exists, but no valid dates were found. "
                    "Date filters are unavailable."
                )

        else:
            st.info(
                "This dataset has no recognized date column. "
                "Date, month and year filters are unavailable."
            )

        st.caption(
            f"Dataset: {selected_home} | "
            f"Filtered records: {len(home_df):,} of {len(source_df):,}"
        )

        total_transactions = len(home_df)
        total_fraud = fraud_count(home_df) if target_col else 0
        current_fraud_rate = fraud_rate(home_df) if target_col else 0

        total_amount = (
            pd.to_numeric(home_df[amount_col], errors="coerce").sum()
            if amount_col else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("📄 Total Transactions", f"{total_transactions:,}")
        k2.metric("🚨 Total Fraud", f"{total_fraud:,}")
        k3.metric("⚠️ Fraud Rate", f"{current_fraud_rate:.2f}%")
        k4.metric(
            "💰 Total Amount",
            f"{total_amount:,.2f}" if amount_col else "N/A"
        )

        st.divider()
        st.subheader("📊 Fraud & Transaction Overview")

        if target_col:
            fraud_total = fraud_count(home_df)
            normal_total = normal_count(home_df)

            left, right = st.columns(2)

            with left:
                status_df = pd.DataFrame({
                    "Transaction Status": ["Normal", "Fraud"],
                    "Transactions": [normal_total, fraud_total]
                })

                fig = px.pie(
                    status_df,
                    names="Transaction Status",
                    values="Transactions",
                    hole=0.55,
                    title="Fraud vs Normal Transactions",
                    color="Transaction Status",
                    color_discrete_map={
                        "Normal": "#16a085",
                        "Fraud": "#e74c3c"
                    }
                )

                show_chart(fig)

            with right:
                if date_col and home_df[date_col].notna().any():
                    daily = (
                        home_df.dropna(subset=[date_col])
                        .groupby(home_df[date_col].dt.floor("D"))
                        .size()
                        .reset_index(name="Transactions")
                    )

                    daily.columns = ["Date", "Transactions"]

                    fig = px.line(
                        daily,
                        x="Date",
                        y="Transactions",
                        markers=True,
                        title="Daily Transactions"
                    )

                    show_chart(fig)

                else:
                    st.info(
                        "Daily transaction trend requires a valid date column."
                    )

            if date_col and home_df[date_col].notna().any():
                trend_data = home_df.dropna(subset=[date_col]).copy()
                trend_data["_fraud_flag"] = fraud_mask(trend_data).astype(int)

                monthly = (
                    trend_data.set_index(date_col)["_fraud_flag"]
                    .resample("MS")
                    .sum()
                    .reset_index()
                )

                monthly.columns = ["Month", "Fraud Cases"]

                fig = px.line(
                    monthly,
                    x="Month",
                    y="Fraud Cases",
                    markers=True,
                    title="Monthly Fraud Trend"
                )

                show_chart(fig)

                yearly = (
                    trend_data.assign(
                        Year=trend_data[date_col].dt.year
                    )
                    .groupby("Year")["_fraud_flag"]
                    .sum()
                    .reset_index(name="Fraud Cases")
                )

                fig = px.bar(
                    yearly,
                    x="Year",
                    y="Fraud Cases",
                    title="Yearly Fraud Trend",
                    color="Fraud Cases",
                    color_continuous_scale="Reds"
                )

                show_chart(fig)

        if amount_col:
            amount_values = pd.to_numeric(
                home_df[amount_col],
                errors="coerce"
            ).dropna()

            if not amount_values.empty:
                st.subheader("💰 Transaction Amount Analysis")

                left, right = st.columns(2)

                with left:
                    fig = px.histogram(
                        amount_values,
                        x=amount_values,
                        nbins=40,
                        title="Transaction Amount Distribution"
                    )

                    show_chart(fig)

                with right:
                    amount_summary = pd.DataFrame({
                        "Metric": [
                            "Minimum", "Average", "Median",
                            "Maximum", "Total"
                        ],
                        "Amount": [
                            amount_values.min(),
                            amount_values.mean(),
                            amount_values.median(),
                            amount_values.max(),
                            amount_values.sum()
                        ]
                    })

                    st.dataframe(
                        amount_summary,
                        use_container_width=True,
                        hide_index=True
                    )

        transaction_type_col = next(
            (
                c for c in [
                    "TransactionType", "Merchant_Category",
                    "Payment_Method", "Device_Type", "channel"
                ]
                if c in home_df.columns
            ),
            None
        )

        location_col = next(
            (
                c for c in ["Location", "location", "MerchantID"]
                if c in home_df.columns
            ),
            None
        )

        if transaction_type_col or location_col:
            st.subheader("🧭 Transaction Category & Location Analysis")

            chart_cols = st.columns(2)

            if transaction_type_col:
                with chart_cols[0]:
                    type_counts = (
                        home_df[transaction_type_col]
                        .astype(str)
                        .value_counts()
                        .head(15)
                        .rename_axis(transaction_type_col)
                        .reset_index(name="Transactions")
                    )

                    fig = px.bar(
                        type_counts,
                        x=transaction_type_col,
                        y="Transactions",
                        title=f"Transactions by {transaction_type_col}",
                        color="Transactions",
                        color_continuous_scale="Blues"
                    )

                    show_chart(fig)

            if location_col:
                with chart_cols[1]:
                    location_counts = (
                        home_df[location_col]
                        .astype(str)
                        .value_counts()
                        .head(15)
                        .rename_axis(location_col)
                        .reset_index(name="Transactions")
                    )

                    fig = px.bar(
                        location_counts,
                        x=location_col,
                        y="Transactions",
                        title=f"Transactions by {location_col}",
                        color="Transactions",
                        color_continuous_scale="Teal"
                    )

                    show_chart(fig)

        st.divider()
        st.subheader("📋 Filtered Transaction Records")

        show_table(home_df)

        st.download_button(
            "⬇️ Download Filtered Transactions",
            data=home_df.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_filtered_transactions.csv",
            mime="text/csv",
            key="home_filtered_transactions_csv"
        )

    else:
        st.warning(
            "No datasets found. Upload a CSV from Data Upload & Reports."
        )


# =========================================================
# 7. EXECUTIVE DASHBOARD
# =========================================================

elif page == "📊 Executive Dashboard":

    st.title("🛡️ FraudShield AI")
    st.subheader("Enterprise Fraud Detection & Risk Intelligence Platform")
    st.caption("Machine Learning • Risk Scoring • Fraud Analytics")

    selected = dataset_selector()

    if selected:
        df = datasets[selected]
        target = target_column(df)
        amount = amount_column(df)
        date = date_column(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📄 Transactions", f"{len(df):,}")
        c2.metric("🚨 Fraud Cases", f"{fraud_count(df):,}")
        c3.metric("⚠️ Fraud Rate", f"{fraud_rate(df):.2f}%")
        c4.metric("✅ Normal Cases", f"{normal_count(df):,}")

        st.divider()
        st.subheader("📊 Transaction Intelligence")

        left, right = st.columns(2)

        with left:
            if target:
                fig = px.pie(
                    names=["Normal Transactions", "Fraud Cases"],
                    values=[normal_count(df), fraud_count(df)],
                    hole=0.55,
                    title="Fraud vs Normal Transactions",
                    color_discrete_sequence=["#16a085", "#e74c3c"]
                )

                show_chart(fig)

            else:
                st.info("Fraud target column not available.")

        with right:
            if amount:
                amount_data = pd.to_numeric(
                    df[amount],
                    errors="coerce"
                ).dropna()

                fig = px.histogram(
                    x=amount_data,
                    nbins=40,
                    title="Transaction Amount Distribution",
                    color_discrete_sequence=["#38bdf8"]
                )

                show_chart(fig)

            else:
                st.info("Amount column not available.")

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

            trend_df = trend_df.dropna()

            if not trend_df.empty:
                trend = (
                    trend_df.set_index(date)
                    .resample("D")[target]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    trend,
                    x=date,
                    y=target,
                    markers=True,
                    title="Daily Fraud Transactions"
                )

                show_chart(fig)

        if target:
            category = next(
                (
                    c for c in [
                        "Merchant_Category", "TransactionType",
                        "Payment_Method", "Device_Type", "Location"
                    ]
                    if c in df.columns
                ),
                None
            )

            if category:
                fraud_records = df[fraud_mask(df)]

                counts = (
                    fraud_records[category]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .rename_axis(category)
                    .reset_index(name="Fraud Cases")
                )

                fig = px.bar(
                    counts,
                    x=category,
                    y="Fraud Cases",
                    color="Fraud Cases",
                    title=f"Fraud Cases by {category}",
                    color_continuous_scale="Blues"
                )

                show_chart(fig)

        st.subheader("📋 Transaction Records")

        fraud_only = st.checkbox(
            "Show only fraud transactions",
            key="exec_fraud_only"
        )

        display_df = (
            df[fraud_mask(df)]
            if fraud_only and target
            else df
        )

        show_table(display_df)

        st.download_button(
            "⬇️ Download Transaction CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_transactions.csv",
            mime="text/csv"
        )

        report_download_button(
            df, selected, "Executive Dashboard Report"
        )


# =========================================================
# 8. DATASET OVERVIEW
# =========================================================

elif page == "📁 Dataset Overview":

    st.title("📁 Dataset Overview")

    selected = dataset_selector()

    if selected:
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

        report_download_button(
            df, selected, "Dataset Overview Report"
        )


# =========================================================
# 9. EDA DASHBOARD
# =========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    selected = dataset_selector()

    if selected:
        df = datasets[selected]

        numeric = df.select_dtypes(include=np.number)
        categorical = df.select_dtypes(
            include=["object", "category"]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Numerical Columns", len(numeric.columns))
        c2.metric("Categorical Columns", len(categorical.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        if not numeric.empty:
            st.subheader("📊 Numerical Statistical Analysis")

            st.dataframe(
                numeric.describe().T,
                use_container_width=True
            )

            st.subheader("🔥 Correlation Heatmap")

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

            show_chart(fig, 650)

            st.subheader("📉 Numerical Feature Distribution")

            selected_num = st.selectbox(
                "Select Numerical Feature",
                numeric.columns
            )

            fig = px.histogram(
                df,
                x=selected_num,
                nbins=40,
                marginal="box",
                title=f"Distribution of {selected_num}"
            )

            show_chart(fig)

            fig = px.box(
                df,
                y=selected_num,
                title=f"Box Plot of {selected_num}"
            )

            show_chart(fig)

        st.subheader("🧩 Missing Values Analysis")

        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            color="Missing Values",
            title="Missing Values by Column",
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
                .rename_axis(selected_col)
                .reset_index(name="Count")
            )

            fig = px.bar(
                counts,
                x=selected_col,
                y="Count",
                color="Count",
                title=f"Distribution of {selected_col}"
            )

            show_chart(fig)

        report_download_button(
            df, selected, "EDA Analysis Report"
        )


# =========================================================
# 10. FEATURE ENGINEERING
# =========================================================

elif page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    selected = dataset_selector()

    if selected:
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

        st.subheader("Feature Information")

        feature_info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            feature_info,
            use_container_width=True,
            hide_index=True
        )

        if numeric:
            selected_feature = st.selectbox(
                "Select feature for transformation preview",
                numeric
            )

            transformed = pd.to_numeric(
                df[selected_feature],
                errors="coerce"
            )

            st.write(
                "Log transformation preview (positive values only)"
            )

            log_values = np.log1p(
                transformed.clip(lower=0)
            )

            preview = pd.DataFrame({
                "Original": transformed.head(100),
                "Log Transformed": log_values.head(100)
            })

            fig = px.scatter(
                preview,
                x="Original",
                y="Log Transformed",
                title="Original vs Log Transformed Values"
            )

            show_chart(fig)

            st.download_button(
                "⬇️ Download Feature Preview",
                data=preview.to_csv(index=False).encode("utf-8"),
                file_name="feature_transformation_preview.csv",
                mime="text/csv"
            )

        st.info(
            "This page provides feature inspection and transformation "
            "previews. Original dataset is not modified."
        )


# =========================================================
# 11. FRAUD INTELLIGENCE
# =========================================================

elif page == "🕵️ Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence")

    selected = dataset_selector()

    if selected:
        df = datasets[selected]
        target = target_column(df)

        if target:
            fraud_records = df[fraud_mask(df)]

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Cases", len(fraud_records))
            c2.metric("Normal Cases", normal_count(df))
            c3.metric("Fraud Rate", f"{fraud_rate(df):.2f}%")

            amount = amount_column(df)

            if amount:
                fig = px.histogram(
                    fraud_records,
                    x=amount,
                    nbins=35,
                    marginal="box",
                    title="Fraud Transaction Amount Distribution",
                    color_discrete_sequence=["#ef5350"]
                )

                show_chart(fig)

            category = next(
                (
                    c for c in [
                        "Merchant_Category", "TransactionType",
                        "Payment_Method", "Device_Type", "Location"
                    ]
                    if c in df.columns
                ),
                None
            )

            if category:
                counts = (
                    fraud_records[category]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .rename_axis(category)
                    .reset_index(name="Fraud Cases")
                )

                fig = px.bar(
                    counts,
                    x=category,
                    y="Fraud Cases",
                    color="Fraud Cases",
                    title=f"Fraud by {category}",
                    color_continuous_scale="Reds"
                )

                show_chart(fig)

            st.subheader("Fraud Transaction Records")
            show_table(fraud_records)

            st.download_button(
                "⬇️ Download Fraud Records",
                data=fraud_records.to_csv(index=False).encode("utf-8"),
                file_name="fraud_records.csv",
                mime="text/csv"
            )

        else:
            st.warning("Fraud target column not found.")


# =========================================================
# 12. MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    rows = []

    for label, filename in MODEL_FILES.items():
        saved = load_model_file(filename)

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
            "Model": label,
            "Model File": filename,
            "Algorithm": type(model).__name__,
            "Threshold": threshold,
            "Number of Features": len(features)
        })

    if rows:
        comparison = pd.DataFrame(rows)

        st.metric("Available Models", len(comparison))

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="Number of Features",
            color="Algorithm",
            title="Model Feature Count"
        )

        show_chart(fig)

        st.download_button(
            "⬇️ Download Model Information",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    else:
        st.warning("No model files found in models folder.")


# =========================================================
# 13. FRAUD PREDICTION
# =========================================================

elif page == "🚨 Fraud Prediction":

    st.title("🚨 Fraud Prediction")

    selected_model = st.selectbox(
        "Select Prediction Model",
        list(MODEL_FILES.keys())
    )

    saved = load_model_file(MODEL_FILES[selected_model])

    if saved is None:
        st.error("Selected model file was not found.")

    elif not isinstance(saved, dict):
        st.error(
            "Saved model must contain model, threshold and features."
        )

    else:
        model = get_model(saved)
        features = get_features(saved, model)
        threshold = saved.get("threshold", 0.5)

        if model is None or not features:
            st.error("Model or feature list is missing.")

        else:
            st.write("Algorithm:", type(model).__name__)
            st.write("Saved threshold:", threshold)

            st.caption(
                "Enter values in the same format used during model training."
            )

            with st.form("fraud_prediction_form"):
                values = {}
                input_columns = st.columns(2)

                categorical_words = [
                    "location", "merchant_category",
                    "transactiontype", "payment_method",
                    "device_type", "suspicious_keyword",
                    "channel", "promo_id"
                ]

                for i, feature in enumerate(features):
                    with input_columns[i % 2]:
                        if any(
                            word in feature.lower()
                            for word in categorical_words
                        ):
                            values[feature] = st.text_input(feature)

                        else:
                            values[feature] = st.number_input(
                                feature,
                                value=0.0,
                                format="%.5f"
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

                    if hasattr(model, "predict_proba"):
                        probability = float(
                            model.predict_proba(input_df)[0][1]
                        )

                    else:
                        st.error(
                            "This model does not support predict_proba."
                        )
                        probability = None

                    if probability is not None:
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
                    st.error(f"Prediction error: {error}")

                    st.info(
                        "Check input data types and encoding against "
                        "the original model training pipeline."
                    )


# =========================================================
# 14. FRAUD ANALYTICS
# =========================================================

elif page == "📊 Fraud Analytics":

    st.title("📊 Fraud Analytics")

    selected = dataset_selector()

    if selected:
        df = datasets[selected]
        target = target_column(df)
        amount = amount_column(df)

        if target:
            fraud_records = df[fraud_mask(df)]

            c1, c2, c3 = st.columns(3)

            c1.metric("Fraud Transactions", len(fraud_records))
            c2.metric("Normal Transactions", normal_count(df))
            c3.metric(
                "Fraud Percentage",
                f"{fraud_rate(df):.2f}%"
            )

            if amount:
                fig = px.box(
                    df,
                    x=target,
                    y=amount,
                    color=target,
                    title="Transaction Amount by Fraud Class"
                )

                show_chart(fig)

                fig = px.violin(
                    df,
                    x=target,
                    y=amount,
                    color=target,
                    box=True,
                    title="Fraud vs Normal Amount Distribution"
                )

                show_chart(fig)

            if "Location" in df.columns:
                location_data = (
                    fraud_records["Location"]
                    .astype(str)
                    .value_counts()
                    .head(15)
                    .rename_axis("Location")
                    .reset_index(name="Fraud Count")
                )

                fig = px.bar(
                    location_data,
                    x="Location",
                    y="Fraud Count",
                    color="Fraud Count",
                    title="Fraud by Location",
                    color_continuous_scale="Reds"
                )

                show_chart(fig)

            report_download_button(
                df, selected, "Fraud Analytics Report"
            )

        else:
            st.warning("Fraud target column not found.")


# =========================================================
# 15. DETAILED INVESTIGATION
# =========================================================

elif page == "🔎 Detailed Investigation":

    st.title("🔎 Detailed Investigation")

    selected = dataset_selector()

    if selected:
        df = datasets[selected]
        target = target_column(df)
        amount = amount_column(df)

        search_columns = [
            c for c in [
                "TransactionID", "Transaction_ID",
                "Customer_ID", "receipt_id"
            ]
            if c in df.columns
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
                    filtered[search_col]
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ]

        if fraud_only and target:
            filtered = filtered[fraud_mask(filtered)]

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
            c2.metric(
                "Fraud Records",
                fraud_count(filtered)
            )

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
        show_table(filtered)

        st.download_button(
            "⬇️ Download Investigation Results",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_investigation.csv",
            mime="text/csv"
        )

        report_download_button(
            filtered,
            selected,
            "Investigation Report"
        )


# =========================================================
# 16. EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI")

    selected = st.selectbox(
        "Select Model",
        list(MODEL_FILES.keys())
    )

    saved = load_model_file(MODEL_FILES[selected])

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
                    importance_df.head(20).sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 20 Feature Importances",
                    color="Importance",
                    color_continuous_scale="Blues"
                )

                show_chart(fig, 600)

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Feature Importance",
                    data=importance_df.to_csv(index=False).encode("utf-8"),
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "Feature names do not match importance values."
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
        "This page shows monitoring insights from your loaded historical "
        "dataset. It is not connected to a live transaction stream."
    )

    selected = dataset_selector()

    if selected:
        df = datasets[selected]
        target = target_column(df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Transactions",
            f"{len(df):,}"
        )

        if target:
            c2.metric(
                "Fraud Transactions",
                fraud_count(df)
            )

            c3.metric(
                "Normal Transactions",
                normal_count(df)
            )

            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[
                    normal_count(df),
                    fraud_count(df)
                ],
                hole=0.5,
                title="Transaction Status",
                color_discrete_sequence=[
                    "#16a085", "#e74c3c"
                ]
            )

            show_chart(fig)

        date = date_column(df)

        if date:
            monitoring = df.copy()

            monitoring[date] = pd.to_datetime(
                monitoring[date],
                errors="coerce"
            )

            monitoring = monitoring.dropna(
                subset=[date]
            )

            if not monitoring.empty:
                daily = monitoring.groupby(
                    monitoring[date].dt.date
                ).size().reset_index(
                    name="Transactions"
                )

                daily.columns = [
                    "Date", "Transactions"
                ]

                fig = px.area(
                    daily,
                    x="Date",
                    y="Transactions",
                    title="Daily Transaction Volume"
                )

                show_chart(fig)

        st.subheader("Latest Available Records")
        show_table(df.tail(50))


# =========================================================
# 18. FRAUD FORECASTING
# =========================================================

elif page == "🔮 Fraud Forecasting":

    st.title("🔮 Fraud Forecasting")

    st.info(
        "This page displays historical fraud trends. A future prediction "
        "requires a separately trained forecasting model."
    )

    selected = dataset_selector()

    if selected:
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

            trend_df = trend_df.dropna()

            if not trend_df.empty:
                trend = (
                    trend_df.set_index(date)
                    .resample("D")[target]
                    .sum()
                    .reset_index()
                )

                fig = px.line(
                    trend,
                    x=date,
                    y=target,
                    markers=True,
                    title="Historical Daily Fraud Transactions"
                )

                show_chart(fig)

                st.dataframe(
                    trend,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Historical Trend",
                    data=trend.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_historical_trend.csv",
                    mime="text/csv"
                )

            else:
                st.warning("No valid date records found.")

        else:
            st.warning(
                "Date column or fraud target column not found."
            )


# =========================================================
# 19. GRAPH ANALYTICS
# =========================================================

elif page == "🔗 Graph Analytics":

    st.title("🔗 Graph Analytics")

    st.info(
        "This page explores relationships in transaction data. "
        "It is not a dedicated Graph Neural Network."
    )

    selected = dataset_selector()

    if selected:
        df = datasets[selected]

        relationship_columns = [
            c for c in [
                "Customer_ID", "Merchant_Category",
                "Payment_Method", "Device_Type",
                "Location", "MerchantID"
            ]
            if c in df.columns
        ]

        if relationship_columns:
            selected_col = st.selectbox(
                "Select Relationship Feature",
                relationship_columns
            )

            counts = (
                df[selected_col]
                .astype(str)
                .value_counts()
                .head(20)
                .rename_axis(selected_col)
                .reset_index(name="Transaction Count")
            )

            fig = px.bar(
                counts,
                x=selected_col,
                y="Transaction Count",
                color="Transaction Count",
                title=f"Transaction Relationships by {selected_col}",
                color_continuous_scale="Teal"
            )

            show_chart(fig)

            target = target_column(df)

            if target:
                fraud_counts = (
                    df[fraud_mask(df)][selected_col]
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .rename_axis(selected_col)
                    .reset_index(name="Fraud Count")
                )

                fig = px.bar(
                    fraud_counts,
                    x=selected_col,
                    y="Fraud Count",
                    color="Fraud Count",
                    title=f"Fraud Relationships by {selected_col}",
                    color_continuous_scale="Reds"
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


# =========================================================
# 20. DATA UPLOAD & REPORTS
# =========================================================

elif page == "📤 Data Upload & Reports":

    st.title("📤 Data Upload & Reports")

    st.markdown("""
    Upload your own CSV dataset. It will be available in the dataset
    selector during this app session. The original GitHub dataset
    files will not be overwritten.
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.write("Dataset Preview")
            show_table(uploaded_df)

            st.write("Dataset Information")

            c1, c2, c3 = st.columns(3)

            c1.metric("Rows", f"{len(uploaded_df):,}")
            c2.metric("Columns", uploaded_df.shape[1])

            c3.metric(
                "Missing Values",
                int(uploaded_df.isnull().sum().sum())
            )

            dataset_name = uploaded_file.name

            if st.button("➕ Add Dataset to Dashboard"):
                st.session_state.uploaded_datasets[
                    dataset_name
                ] = uploaded_df

                st.success(
                    "Dataset added. It is now available in the dashboard selectors."
                )

                st.rerun()

        except Exception as error:
            st.error(
                f"Could not read uploaded CSV: {error}"
            )

    st.divider()
    st.subheader("📚 Available Datasets")

    if datasets:
        for name, df in datasets.items():
            with st.expander(name):
                c1, c2, c3 = st.columns(3)

                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", df.shape[1])
                c3.metric("Fraud Cases", fraud_count(df))

                st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                report_download_button(
                    df,
                    name,
                    f"{name} Report"
                )

                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_{name}"
                )

    else:
        st.warning(
            "No datasets are currently available."
        )

    st.divider()
    st.subheader("📄 Full Dataset Report")

    selected = dataset_selector(
        "report_dataset_selector"
    )

    if selected:
        report_download_button(
            datasets[selected],
            selected,
            "FraudShield AI Full Dataset Report"
        )


# =========================================================
# 21. GLOBAL KPI ANALYTICS + OFFLINE AI CHATBOT
# =========================================================

st.divider()
st.header("📊 FraudShield Global Intelligence Center")

if datasets:

    global_dataset = st.selectbox(
        "Choose Dataset for Global Analytics",
        list(datasets.keys()),
        key="global_analytics_dataset"
    )

    global_df = datasets[global_dataset].copy()

    global_target = target_column(global_df)
    global_amount = amount_column(global_df)
    global_date = date_column(global_df)

    total_records = len(global_df)
    total_fraud_cases = fraud_count(global_df)
    total_normal_cases = normal_count(global_df)
    total_missing = int(global_df.isnull().sum().sum())
    total_duplicates = int(global_df.duplicated().sum())

    fraud_percentage = fraud_rate(global_df)

    amount_values = pd.to_numeric(
        global_df[global_amount],
        errors="coerce"
    ).dropna() if global_amount else pd.Series(dtype=float)

    total_transaction_amount = (
        amount_values.sum() if not amount_values.empty else 0
    )

    average_transaction_amount = (
        amount_values.mean() if not amount_values.empty else 0
    )

    # GLOBAL KPI CARDS

    st.subheader("📌 Key Performance Indicators")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("📄 Total Transactions", f"{total_records:,}")
    k2.metric("🚨 Fraud Cases", f"{total_fraud_cases:,}")
    k3.metric("⚠️ Fraud Rate", f"{fraud_percentage:.2f}%")
    k4.metric("✅ Normal Transactions", f"{total_normal_cases:,}")

    k5, k6, k7, k8 = st.columns(4)

    k5.metric("💰 Total Amount", f"{total_transaction_amount:,.2f}")
    k6.metric("💳 Average Amount", f"{average_transaction_amount:,.2f}")
    k7.metric("❓ Missing Values", f"{total_missing:,}")
    k8.metric("🔁 Duplicate Rows", f"{total_duplicates:,}")

    # GLOBAL CHARTS

    st.subheader("📈 Advanced Dataset Visualizations")

    chart_left, chart_right = st.columns(2)

    with chart_left:

        if global_target:

            status_data = pd.DataFrame({
                "Status": ["Normal", "Fraud"],
                "Count": [
                    total_normal_cases,
                    total_fraud_cases
                ]
            })

            fig = px.pie(
                status_data,
                names="Status",
                values="Count",
                hole=0.6,
                title="Fraud vs Normal Distribution",
                color="Status",
                color_discrete_map={
                    "Normal": "#16a085",
                    "Fraud": "#e74c3c"
                }
            )

            show_chart(fig)

        else:
            st.info("Fraud target column not available.")

    with chart_right:

        if not amount_values.empty:

            fig = px.histogram(
                x=amount_values,
                nbins=50,
                title="Transaction Amount Distribution",
                marginal="box"
            )

            show_chart(fig)

        else:
            st.info("Amount column not available.")

    if global_target and global_amount:

        fraud_amount_data = global_df.copy()

        fraud_amount_data["_FraudStatus"] = np.where(
            fraud_mask(fraud_amount_data),
            "Fraud",
            "Normal"
        )

        fraud_amount_data["_Amount"] = pd.to_numeric(
            fraud_amount_data[global_amount],
            errors="coerce"
        )

        fraud_amount_data = fraud_amount_data.dropna(
            subset=["_Amount"]
        )

        if not fraud_amount_data.empty:

            fig = px.box(
                fraud_amount_data,
                x="_FraudStatus",
                y="_Amount",
                color="_FraudStatus",
                title="Fraud vs Normal Amount Comparison"
            )

            show_chart(fig)

    # DATE ANALYSIS

    if global_date:

        date_data = global_df.copy()

        date_data[global_date] = pd.to_datetime(
            date_data[global_date],
            errors="coerce"
        )

        date_data = date_data.dropna(
            subset=[global_date]
        )

        if not date_data.empty:

            date_data["_Date"] = date_data[global_date].dt.date

            daily_data = (
                date_data.groupby("_Date")
                .size()
                .reset_index(name="Transactions")
            )

            fig = px.area(
                daily_data,
                x="_Date",
                y="Transactions",
                title="Daily Transaction Volume"
            )

            show_chart(fig)

            if global_target:

                date_data["_FraudFlag"] = (
                    fraud_mask(date_data).astype(int)
                )

                daily_fraud = (
                    date_data.groupby("_Date")["_FraudFlag"]
                    .sum()
                    .reset_index(name="Fraud Cases")
                )

                fig = px.line(
                    daily_fraud,
                    x="_Date",
                    y="Fraud Cases",
                    markers=True,
                    title="Daily Fraud Trend"
                )

                show_chart(fig)

    # CATEGORY ANALYSIS

    category_columns = [
        col for col in [
            "Merchant_Category",
            "TransactionType",
            "Payment_Method",
            "Device_Type",
            "Location",
            "channel",
            "Channel"
        ]
        if col in global_df.columns
    ]

    if category_columns:

        selected_category = st.selectbox(
            "Select Category for Analysis",
            category_columns,
            key="global_category_analysis"
        )

        category_counts = (
            global_df[selected_category]
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis(selected_category)
            .reset_index(name="Transactions")
        )

        fig = px.bar(
            category_counts,
            x=selected_category,
            y="Transactions",
            color="Transactions",
            title=f"Transactions by {selected_category}",
            color_continuous_scale="Blues"
        )

        show_chart(fig)

    st.divider()

    # =====================================================
    # OFFLINE FRAUDSHIELD AI CHAT ASSISTANT
    # =====================================================

    st.header("🤖 FraudShield AI Chat Assistant")

    st.caption(
        "Ask questions about your selected dataset. "
        "This is an offline, rule-based assistant; no API key is required."
    )

    if "fraudshield_chat_history" not in st.session_state:
        st.session_state.fraudshield_chat_history = []

    # Show chat history

    for message in st.session_state.fraudshield_chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input

    user_question = st.chat_input(
        "Ask about fraud, transactions, amount, dataset..."
    )

    if user_question:

        st.session_state.fraudshield_chat_history.append({
            "role": "user",
            "content": user_question
        })

        question = user_question.lower()

        # Dataset-based answer generation

        if any(word in question for word in [
            "total", "transactions", "records", "rows"
        ]):

            answer = (
                f"📄 **Total Transactions:** {total_records:,}\n\n"
                f"Dataset: {global_dataset}"
            )

        elif any(word in question for word in [
            "fraud rate", "percentage", "fraud percentage"
        ]):

            answer = (
                f"⚠️ **Fraud Rate:** {fraud_percentage:.2f}%\n\n"
                f"Fraud Cases: {total_fraud_cases:,}\n\n"
                f"Normal Cases: {total_normal_cases:,}"
            )

        elif "fraud" in question:

            if global_target:

                answer = (
                    f"🚨 **Fraud Analysis**\n\n"
                    f"- Fraud Cases: {total_fraud_cases:,}\n"
                    f"- Normal Cases: {total_normal_cases:,}\n"
                    f"- Fraud Rate: {fraud_percentage:.2f}%"
                )

            else:

                answer = (
                    "Is dataset mein recognized fraud target column "
                    "nahi mila. Fraud analysis ke liye IsFraud, "
                    "Fraudulent, is_fraud, fraud ya Class column chahiye."
                )

        elif any(word in question for word in [
            "normal", "genuine", "legitimate"
        ]):

            answer = (
                f"✅ **Normal Transactions:** {total_normal_cases:,}"
            )

        elif any(word in question for word in [
            "amount", "money", "value", "average", "total value"
        ]):

            if not amount_values.empty:

                answer = (
                    f"💰 **Transaction Amount Analysis**\n\n"
                    f"- Total Amount: {total_transaction_amount:,.2f}\n"
                    f"- Average Amount: {average_transaction_amount:,.2f}\n"
                    f"- Median Amount: {amount_values.median():,.2f}\n"
                    f"- Maximum Amount: {amount_values.max():,.2f}\n"
                    f"- Minimum Amount: {amount_values.min():,.2f}"
                )

            else:

                answer = "Amount column dataset mein available nahi hai."

        elif any(word in question for word in [
            "missing", "null", "nan", "empty"
        ]):

            answer = (
                f"❓ **Missing Values:** {total_missing:,}\n\n"
                f"Dataset mein {global_df.shape[1]} columns hain."
            )

        elif any(word in question for word in [
            "duplicate", "duplicates"
        ]):

            answer = (
                f"🔁 **Duplicate Rows:** {total_duplicates:,}"
            )

        elif any(word in question for word in [
            "column", "features", "fields"
        ]):

            answer = (
                f"📊 **Dataset Columns ({len(global_df.columns)}):**\n\n"
                + "\n".join(
                    [f"- {col}" for col in global_df.columns]
                )
            )

        elif any(word in question for word in [
            "dataset", "data information", "overview"
        ]):

            answer = (
                f"📁 **Dataset Overview**\n\n"
                f"- Name: {global_dataset}\n"
                f"- Rows: {global_df.shape[0]:,}\n"
                f"- Columns: {global_df.shape[1]}\n"
                f"- Missing Values: {total_missing:,}\n"
                f"- Duplicate Rows: {total_duplicates:,}"
            )

        elif any(word in question for word in [
            "help", "what can you do", "commands"
        ]):

            answer = (
                "🤖 **I can answer these dataset questions:**\n\n"
                "- Total transactions\n"
                "- Total fraud cases\n"
                "- Fraud percentage / fraud rate\n"
                "- Normal transactions\n"
                "- Total and average transaction amount\n"
                "- Missing values\n"
                "- Duplicate rows\n"
                "- Dataset columns and overview\n\n"
                "Example: `What is the fraud rate?`"
            )

        else:

            answer = (
                "I can help analyze your selected dataset. "
                "Try asking: **total transactions**, **fraud cases**, "
                "**fraud rate**, **average amount**, **missing values**, "
                "or **show columns**."
            )

        st.session_state.fraudshield_chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    chat_col1, chat_col2 = st.columns([1, 4])

    with chat_col1:

        if st.button(
            "🗑️ Clear Chat",
            key="clear_fraudshield_chat"
        ):

            st.session_state.fraudshield_chat_history = []
            st.rerun()

    with chat_col2:

        st.caption(
            f"Assistant dataset: {global_dataset}"
        )

else:

    st.info(
        "Upload a CSV dataset from Data Upload & Reports "
        "to activate Global Intelligence and Chat Assistant."
    )

# =========================================================
# 21. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🛡️ <b>FraudShield AI</b><br>
    Enterprise Fraud Detection & Risk Intelligence Platform<br>
    Built with Python • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)