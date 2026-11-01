import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

files = {
    "Credit Card Fraud": "credit_card_fraud_processed.csv",
    "Fraud Detection": "fraud_detection_processed.csv"
}

for name, filename in files.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    df = pd.read_csv(PROCESSED_DIR / filename)

    print("Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nFraud Distribution:")
    print(df["Fraud"].value_counts())

    print("\nDate Column Type:")
    print(df["Transaction_Date"].dtype)

    print("\nFirst 5 Rows:")
    print(df.head())

print("\nPreprocessing validation completed!")