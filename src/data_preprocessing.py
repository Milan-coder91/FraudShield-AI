import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_dataset.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_dataset.csv")

# Dataset 1: Rename columns
df1 = df1.rename(columns={
    "TransactionID": "Transaction_ID",
    "TransactionDate": "Transaction_Date",
    "Amount": "Transaction_Amount",
    "TransactionType": "Transaction_Type",
    "IsFraud": "Fraud"
})

# Dataset 2: Rename target column
df2 = df2.rename(columns={
    "Fraudulent": "Fraud"
})

# Convert date columns
df1["Transaction_Date"] = pd.to_datetime(
    df1["Transaction_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df2["Transaction_Date"] = pd.to_datetime(
    df2["Transaction_Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Add dataset source
df1["Dataset_Source"] = "Credit_Card"
df2["Dataset_Source"] = "Fraud_Detection"

# Save separately
df1.to_csv(PROCESSED_DIR / "credit_card_fraud_processed.csv", index=False)
df2.to_csv(PROCESSED_DIR / "fraud_detection_processed.csv", index=False)

print("Dataset 1 processed shape:", df1.shape)
print("Dataset 2 processed shape:", df2.shape)
print("\nBoth datasets preprocessed and saved successfully!")