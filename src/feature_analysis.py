import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_dataset.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_dataset.csv")

print("\n===== CREDIT CARD FRAUD: FEATURE ANALYSIS =====")

print("\nTransaction Amount Statistics:")
print(df1["Amount"].describe())

print("\nTransaction Type:")
print(df1["TransactionType"].value_counts())

print("\nLocation:")
print(df1["Location"].value_counts())

print("\nAmount by Fraud Status:")
print(df1.groupby("IsFraud")["Amount"].mean())


print("\n===== FRAUD DETECTION: FEATURE ANALYSIS =====")

print("\nTransaction Amount Statistics:")
print(df2["Transaction_Amount"].describe())

print("\nMerchant Category:")
print(df2["Merchant_Category"].value_counts())

print("\nPayment Method:")
print(df2["Payment_Method"].value_counts())

print("\nFraud by International Transaction:")
print(pd.crosstab(df2["Is_International"], df2["Fraudulent"]))

print("\nAmount by Fraud Status:")
print(df2.groupby("Fraudulent")["Transaction_Amount"].mean())

print("\nFeature analysis completed successfully!")