import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Load processed datasets
df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_processed.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_processed.csv")

# Dataset 1: Feature Engineering
df1["Transaction_Date"] = pd.to_datetime(df1["Transaction_Date"])
df1["Transaction_Month"] = df1["Transaction_Date"].dt.month
df1["Transaction_Day"] = df1["Transaction_Date"].dt.day
df1["Transaction_Weekday"] = df1["Transaction_Date"].dt.dayofweek
df1["Is_Weekend"] = (df1["Transaction_Weekday"] >= 5).astype(int)

# Dataset 2: Feature Engineering
df2["Transaction_Date"] = pd.to_datetime(df2["Transaction_Date"])
df2["Transaction_Month"] = df2["Transaction_Date"].dt.month
df2["Transaction_Day"] = df2["Transaction_Date"].dt.day
df2["Transaction_Hour"] = df2["Transaction_Date"].dt.hour
df2["Transaction_Weekday"] = df2["Transaction_Date"].dt.dayofweek
df2["Is_Weekend"] = (df2["Transaction_Weekday"] >= 5).astype(int)

# Save feature-engineered datasets
df1.to_csv(OUTPUT_DIR / "credit_card_fraud_features.csv", index=False)
df2.to_csv(OUTPUT_DIR / "fraud_detection_features.csv", index=False)

print("Credit Card Features Shape:", df1.shape)
print("Fraud Detection Features Shape:", df2.shape)
print("\nFeature engineering completed successfully!")