# 🛡️ FraudShield AI

### AI-Powered Financial Fraud Detection & Transaction Intelligence Platform

<p align="center">
  A Machine Learning-powered fraud detection and transaction monitoring platform built using Python, Random Forest, Scikit-learn, and Streamlit.
</p>

<p align="center">
  <a href="https://github.com/Milan-coder91/FraudShield-AI">GitHub Repository</a> |
  <a href="https://fraudshield-ai-milan.streamlit.app/">Live Dashboard</a>
</p>

---

## 🔗 Project Links

- **GitHub Repository:** https://github.com/Milan-coder91/FraudShield-AI
- **Live Streamlit Dashboard:** https://fraudshield-ai-milan.streamlit.app/

## 📌 Project Overview

**FraudShield AI** is a Machine Learning-based financial fraud detection and transaction intelligence platform designed to identify potentially fraudulent transactions.

The project uses two independent financial fraud detection datasets and trained Random Forest classification models to predict whether a transaction is normal or potentially fraudulent.

It also provides an interactive Streamlit dashboard that allows users to explore transaction data, analyze fraud patterns, generate predictions, monitor fraud-related metrics, and investigate suspicious transactions.

The platform combines Machine Learning, data analytics, interactive visualizations, and an offline rule-based chat assistant in one application.

## 🎯 Project Objectives

- Detect potentially fraudulent financial transactions.
- Analyze transaction patterns and fraud distribution.
- Perform data cleaning and preprocessing.
- Apply feature engineering techniques.
- Train and evaluate Random Forest classification models.
- Handle imbalanced datasets using SMOTE.
- Optimize classification decision thresholds.
- Compare different model performances.
- Build an interactive fraud detection dashboard.
- Provide transaction-level fraud predictions and analytics.
- Support fraud investigation through interactive visualizations.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | SMOTE and class imbalance handling |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| ReportLab | PDF report generation |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
FraudShield_AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── reports/
│
├── src/
│   ├── app.py
│   ├── dataset_check.py
│   ├── dataset_analysis.py
│   ├── dataset_visualization.py
│   ├── feature_analysis.py
│   ├── correlation_analysis.py
│   ├── data_preprocessing.py
│   ├── preprocessing_validation.py
│   ├── eda_report.py
│   ├── feature_engineering.py
│   ├── fraudguard_feature_check.py
│   ├── fraudguard_model_training.py
│   ├── fraudguard_smote_training.py
│   ├── fraudguard_threshold_tuning.py
│   ├── fraudguard_model_comparison.py
│   ├── fraudguard_final_evaluation.py
│   └── fraudguard_results_visualization.py
│
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## 📊 Datasets

The project uses two independent fraud detection datasets.

### 1. Credit Card Fraud Dataset

| Dataset Information | Value |
|---|---:|
| Total Transactions | 100,000 |
| Original Columns | 7 |
| Normal Transactions | 99,000 |
| Fraudulent Transactions | 1,000 |
| Fraud Rate | 1% |

### 2. Fraud Detection Dataset

| Dataset Information | Value |
|---|---:|
| Total Transactions | 5,000 |
| Original Columns | 14 |
| Normal Transactions | 4,518 |
| Fraudulent Transactions | 482 |
| Fraud Rate | 9.64% |

Both datasets are analyzed and processed independently, with separate trained models used for their respective prediction workflows.

## ⚙️ Machine Learning Workflow

The project follows a structured Machine Learning pipeline:

1. Dataset Loading and Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Preprocessing
4. Missing Value and Duplicate Analysis
5. Feature Engineering
6. Random Forest Model Training
7. SMOTE-Based Model Training
8. Decision Threshold Tuning
9. Model Comparison
10. Final Model Evaluation
11. Results Visualization
12. Model Serialization
13. Streamlit Dashboard Integration
14. Interactive Fraud Prediction

## 🤖 Machine Learning Models

The project explores the following approaches:

- **Random Forest Classifier:** Classification of normal and potentially fraudulent transactions.
- **SMOTE-Based Random Forest:** Training with oversampling to address class imbalance.
- **Pipeline-Based Models:** Combining preprocessing and classification steps.
- **Threshold Optimization:** Selecting decision thresholds to balance fraud detection and false positives.

The dashboard uses saved Pipeline-based Random Forest models for transaction prediction.

## 📈 Final Model Performance

The following results were obtained from the final test evaluation.

| Evaluation Metric | Credit Card Fraud | Fraud Detection |
|---|---:|---:|
| Accuracy | 31.99% | 83.60% |
| Precision | 1.01% | 33.00% |
| Recall | 69.00% | 68.75% |
| F1 Score | 0.020 | 0.446 |

### Performance Observations

**Credit Card Fraud Dataset**
- The model identifies 69% of fraudulent transactions in the test set.
- Precision is very low at 1.01%, indicating a high number of false-positive predictions.
- Further model improvement and threshold optimization are required.

**Fraud Detection Dataset**
- The model achieves 83.60% accuracy.
- Fraud recall is 68.75%.
- Precision is 33%, indicating scope for reducing false-positive predictions.

**Important:** Accuracy alone is not sufficient for evaluating fraud detection models, particularly when datasets are imbalanced. Precision, recall, F1-score, and the confusion matrix are also important.

## 🖥️ Streamlit Dashboard Features

FraudShield AI includes an interactive dashboard with multiple sections for transaction analysis, fraud detection, and investigation.

### 📊 1. Executive Dashboard
- Overall transaction statistics.
- Fraud and normal transaction KPIs.
- Fraud rate and transaction amount insights.
- Interactive visual summaries.

### 📁 2. Dataset Overview
- Dataset selection.
- Dataset dimensions and column information.
- Data type and missing value inspection.
- Dataset preview.

### 📈 3. EDA Dashboard
- Fraud distribution analysis.
- Transaction amount analysis.
- Interactive charts and visualizations.
- Transaction pattern exploration.

### ⚙️ 4. Feature Engineering
- Feature inspection.
- Numerical and categorical feature analysis.
- Prepared feature information for model workflows.

### 🕵️ 5. Fraud Intelligence
- Fraud pattern exploration.
- Suspicious transaction analysis.
- Fraud-related insights.

### 🤖 6. Model Comparison
- Model performance comparison.
- Accuracy, precision, recall, and F1-score analysis.
- Evaluation results visualization.

### 🚨 7. Fraud Prediction
- Transaction information input.
- Fraud probability calculation.
- Model decision threshold display.
- FRAUD or NORMAL prediction.

### 📊 8. Fraud Analytics
- Fraud distribution analysis.
- Transaction analytics.
- Interactive fraud charts.

### 🔎 9. Detailed Investigation
- Transaction-level investigation.
- Suspicious transaction exploration.
- Detailed dataset filtering.

### 🧠 10. Explainable AI
- Model prediction interpretation.
- Feature-related insights where supported by the selected model.

### 📡 11. Real-Time Monitoring
- Interactive transaction monitoring interface.
- Fraud-related monitoring metrics.

### 🔮 12. Fraud Forecasting
- Fraud and transaction trend exploration.
- Forecast-related visual analysis.

### 🔗 13. Graph Analytics
- Relationship and network-based fraud exploration.

### 📤 14. Data Upload & Reports
- Dataset upload interface.
- Data analysis and reporting.
- Downloadable reports where supported.

### 🏠 15. Home Dashboard
- Project introduction.
- Dashboard navigation.
- Overview of FraudShield AI.

## 💬 FraudShield AI Chat Assistant

The dashboard also includes an **offline, rule-based chat assistant** that helps users explore information about the selected dataset without requiring an external API key.

Users can ask questions such as:

- Show total transactions.
- How many fraud cases are there?
- What is the fraud rate?
- How many normal transactions?
- What is the average transaction amount?
- Show missing values.
- How many duplicate rows?
- Show dataset columns.
- Show dataset overview.

The assistant responds using available dataset information and predefined question-handling rules.

**Note:** This is a rule-based dataset assistant, not a generative AI chatbot.

## 📊 Global KPI Cards & Interactive Visualizations

The dashboard provides dataset-level metrics and visual analytics, including:

- Total Transactions
- Fraud Cases
- Fraud Rate
- Normal Transactions
- Total Transaction Amount
- Average Transaction Amount
- Missing Values
- Duplicate Rows

Interactive visualizations include:

- Fraud vs Normal Distribution
- Transaction Amount Histogram
- Transaction Amount Box Plot
- Daily Transaction Trends
- Daily Fraud Trends
- Category-Based Transaction Analysis

## 🚀 Try the Live Application

Explore the deployed FraudShield AI dashboard:

**https://fraudshield-ai-milan.streamlit.app/**

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Milan-coder91/FraudShield-AI.git
```

### 2. Navigate to the Project Folder

```bash
cd FraudShield-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Dashboard

```bash
streamlit run src/app.py
```

The application will open in your default web browser.

## 📁 Generated Outputs

The project generates or uses the following outputs:

- Processed datasets
- Exploratory Data Analysis reports
- Correlation heatmaps
- Fraud distribution visualizations
- Trained Machine Learning models
- SMOTE model results
- Model comparison reports
- Final evaluation metrics
- Evaluation graphs
- Interactive dashboard analytics
- Downloadable reports

## ⚠️ Disclaimer

FraudShield AI is an educational and demonstration project.

Its predictions are based on trained Machine Learning models and are not guaranteed to be correct. The system may produce false-positive and false-negative predictions.

It should not be used as the sole basis for real financial, banking, or security decisions.

## 👨‍💻 Author

**Milan Kumar**

Data Analysis | Data Science | Machine Learning | Python

---

⭐ If you find this project useful or interesting, consider giving the repository a star!

**Project:** FraudShield AI  
**Category:** Machine Learning | Data Science | Financial Analytics  
**Application:** Financial Fraud Detection & Transaction Monitoring

**GitHub:** https://github.com/Milan-coder91/FraudShield-AI

**Live Demo:** https://fraudshield-ai-milan.streamlit.app/
