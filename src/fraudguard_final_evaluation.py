
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\nModel: Pipeline Random Forest")
    print("Threshold:", threshold)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    all_results.append({
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
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "True_Negatives": cm[0][0],
        "False_Positives": cm[0][1],
        "False_Negatives": cm[1][0],
        "True_Positives": cm[1][1]
    })

results_df = pd.DataFrame(all_results)

results_path = OUTPUT_DIR / "fraudguard_final_results.csv"

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("FINAL EVALUATION SUMMARY")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nResults saved:", results_path)
print("\nFraudGuard Final Evaluation Completed!")
