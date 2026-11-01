import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

datasets = {
    "Credit Card Fraud": (
        "credit_card_fraud_features.csv",
        "credit_card_fraud_smote_model.pkl"
    ),
    "Fraud Detection": (
        "fraud_detection_features.csv",
        "fraud_detection_smote_model.pkl"
    )
}

for name, (filename, model_filename) in datasets.items():

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

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

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train, y_train
    )

    print("\nAfter SMOTE:")
    print(y_train_resampled.value_counts())

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(
        {
            "model": model,
            "features": X.columns.tolist()
        },
        MODEL_DIR / model_filename
    )

    print("\nSaved:", model_filename)

print("\nSMOTE Training Completed!")