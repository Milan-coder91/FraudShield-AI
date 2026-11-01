import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_dataset.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_dataset.csv")

print("\nCREDIT CARD FRAUD DATASET")
print(df1.describe(include="all"))
print("\nFraud Percentage:")
print(df1["IsFraud"].value_counts(normalize=True) * 100)

print("\nFRAUD DETECTION DATASET")
print(df2.describe(include="all"))
print("\nFraud Percentage:")
print(df2["Fraudulent"].value_counts(normalize=True) * 100)

print("\nDataset analysis completed!")