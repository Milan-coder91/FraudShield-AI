import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

datasets = {
    "Credit Card Fraud": (
        "credit_card_fraud_features.csv",
        "credit_card_fraud_model.pkl"
    ),
    "Fraud Detection": (
        "fraud_detection_features.csv",
        "fraud_detection_model.pkl"
    )
}

thresholds = [
    0.001, 0.003, 0.005, 0.01, 0.02,
    0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
]

for name, (filename, model_filename) in datasets.items():

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    df = pd.read_csv(DATA_DIR / filename)

    y = df["Fraud"]

    X = df.drop(
        columns=[
            "Fraud",
            "Transaction_ID",
            "TransactionDate",
            "Transaction_Date",
            "Dataset_Source"
        ],
        errors="ignore"
    )

    # Same 60/20/20 split used during model training
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Load latest Pipeline model
    saved = joblib.load(MODEL_DIR / model_filename)

    model = saved["model"]

    # Threshold tuning only on validation data
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    print("\nValidation Threshold Results:")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[
        results_df["F1 Score"].idxmax()
    ]

    best_threshold = float(best_row["Threshold"])

    print("\nBest Validation Threshold:", best_threshold)
    print("Validation Precision:", best_row["Precision"])
    print("Validation Recall:", best_row["Recall"])
    print("Validation F1 Score:", best_row["F1 Score"])

    # Final evaluation on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nFinal Test Results:")
    print(
        "Precision:",
        precision_score(y_test, test_predictions, zero_division=0)
    )
    print(
        "Recall:",
        recall_score(y_test, test_predictions, zero_division=0)
    )
    print(
        "F1 Score:",
        f1_score(y_test, test_predictions, zero_division=0)
    )

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

print("\nThreshold Tuning Completed!")