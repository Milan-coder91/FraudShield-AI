
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

for name, (filename, model_filename) in datasets.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / filename)

    y = df["Fraud"].astype(int)

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

    # Identify column types
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    # 60% training, 20% validation, 20% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Numerical preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    # Complete model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Select threshold using validation data only
    val_probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        val_probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()
    best_threshold = float(thresholds[best_index])

    # Final testing on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]

    y_pred = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nSelected Threshold:", round(best_threshold, 4))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save complete pipeline
    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "features": X.columns.tolist()
        },
        MODEL_DIR / model_filename
    )

    print("\nModel saved:", model_filename)

print("\nFraudGuard Model Training Completed!")
