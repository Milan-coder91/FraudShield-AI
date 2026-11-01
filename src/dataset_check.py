
import pandas as pd
from pathlib import Path

# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

datasets = {
    "Credit Card Fraud": DATA_DIR / "credit_card_fraud_dataset.csv",
    "Fraud Detection": DATA_DIR / "fraud_detection_dataset.csv"
}

for name, path in datasets.items():
    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    df = pd.read_csv(path)

    print("Shape (Rows, Columns):", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    # Find possible fraud target column
    target_cols = [
        col for col in df.columns
        if any(word in col.lower()
               for word in ["fraud", "isfraud", "class", "target"])
    ]

    for col in target_cols:
        print(f"\nTarget Distribution: {col}")
        print(df[col].value_counts(dropna=False))

print("\nDataset checking completed successfully!")
