# 🛡️ FraudShield AI

### AI-Powered Financial Fraud Detection & Transaction Monitoring System

<p align="center">
  A Machine Learning project designed to identify potentially fraudulent financial transactions using Random Forest classification and an interactive Streamlit dashboard.
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

**FraudShield AI** is a Machine Learning-based fraud detection system designed to identify potentially fraudulent financial transactions using transaction data.

The project uses two independent fraud detection datasets and Random Forest classification models to predict whether a transaction is normal or potentially fraudulent.

It also provides an interactive **Streamlit Dashboard** where users can enter transaction details, view fraud probabilities, and receive real-time predictions.

## 🎯 Project Objectives

- Detect potentially fraudulent financial transactions.
- Analyze transaction patterns and fraud distribution.
- Perform data preprocessing and feature engineering.
- Train and evaluate Machine Learning models.
- Handle imbalanced datasets using SMOTE.
- Optimize classification thresholds.
- Develop an interactive fraud detection dashboard.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | SMOTE |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
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

The project uses two independent datasets.

### 1. Credit Card Fraud Dataset

- Total transactions: 100,000
- Original columns: 7
- Normal transactions: 99,000
- Fraudulent transactions: 1,000
- Fraud rate: 1%

### 2. Fraud Detection Dataset

- Total transactions: 5,000
- Original columns: 14
- Normal transactions: 4,518
- Fraudulent transactions: 482
- Fraud rate: 9.64%

## ⚙️ Project Workflow

1. Dataset Loading and Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Preprocessing
4. Feature Engineering
5. Random Forest Model Training
6. SMOTE-Based Model Training
7. Threshold Tuning
8. Model Comparison
9. Final Model Evaluation
10. Results Visualization
11. Streamlit Dashboard Development

## 🤖 Machine Learning Models

The project explores:

- Random Forest Classifier
- SMOTE-based Random Forest training
- Pipeline-based preprocessing and classification
- Validation-based decision threshold selection

The final dashboard uses the saved Pipeline-based Random Forest models.

## 📈 Final Model Performance

The following results were obtained from the final test evaluation.

| Evaluation Metric | Credit Card Fraud | Fraud Detection |
|---|---:|---:|
| Accuracy | 31.99% | 83.60% |
| Precision | 1.01% | 33.00% |
| Recall | 69.00% | 68.75% |
| F1 Score | 0.020 | 0.446 |

### Performance Observations

- **Credit Card Fraud:** The model detects 69% of fraudulent transactions, but its precision is low and it produces a high number of false positives.
- **Fraud Detection:** The model achieves 83.60% accuracy and 68.75% fraud recall on the test dataset.

**Note:** The Credit Card Fraud model requires further improvement before practical deployment. Model predictions are not guaranteed to be correct.

## 🖥️ Streamlit Dashboard Features

FraudShield AI provides an interactive dashboard with:

- Selection between two fraud detection datasets.
- Transaction information input.
- Fraud probability calculation.
- Model decision threshold display.
- FRAUD or NORMAL prediction.
- Separate trained models for each dataset.

### 🚀 Try the Live Application

Access the deployed dashboard here:

**https://fraudshield-ai-milan.streamlit.app/**

## 🚀 Installation & Setup

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

### 6. Run the Dashboard

```bash
streamlit run src/app.py
```

The dashboard will open in your browser.

## 📁 Generated Outputs

The project generates:

- EDA reports
- Correlation heatmaps
- Fraud distribution charts
- Processed datasets
- Trained Machine Learning models
- Model comparison reports
- Final evaluation results
- Final evaluation graphs

## ⚠️ Disclaimer

FraudShield AI is an educational and demonstration project. Its predictions are based on trained Machine Learning models and are not a guarantee of transaction safety.

It should not be used as the sole basis for real financial or banking decisions.

## 👨‍💻 Author

**Milan Kumar**

Data Analysis | Data Science | Machine Learning | Python

---

⭐ If you find this project interesting, consider giving the repository a star!

**Project:** FraudShield AI  
**Category:** Machine Learning & Data Science and Analysis  
**Application:** Financial Fraud Detection

**GitHub:** https://github.com/Milan-coder91/FraudShield-AI  
**Live Demo:** https://fraudshield-ai-milan.streamlit.app/
