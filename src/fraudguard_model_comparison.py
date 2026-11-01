
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

all_results = []

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

    # Same 60/20/20 split used during training
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
    threshold = saved["threshold"]

    # Test predictions using saved threshold
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {
        "Dataset": name,
        "Model": "Pipeline Random Forest",
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    all_results.append(result)

    print("\nModel:", result["Model"])
    print("Threshold:", result["Threshold"])
    print("Accuracy:", result["Accuracy"])
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("F1 Score:", result["F1 Score"])

comparison = pd.DataFrame(all_results)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))

# Save comparison report
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT_DIR / "fraudguard_model_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
print("\nModel Comparison Completed!")
