import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

datasets = {
    "Credit Card Fraud Features": (
        "credit_card_fraud_features.csv",
        ["Transaction_Month", "Transaction_Day",
         "Transaction_Weekday", "Is_Weekend"]
    ),
    "Fraud Detection Features": (
        "fraud_detection_features.csv",
        ["Transaction_Month", "Transaction_Day",
         "Transaction_Hour", "Transaction_Weekday",
         "Is_Weekend"]
    ),
}

for name, (filename, required_features) in datasets.items():
    df = pd.read_csv(DATA_DIR / filename)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Shape:", df.shape)
    print("Missing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())

    print("\nFraud Distribution:")
    print(df["Fraud"].value_counts())

    print("\nFeature Check:")
    for col in required_features:
        print(f"{col}: {'Available' if col in df.columns else 'Missing'}")

print("\nFraudGuard Feature Check Completed!")