import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent

df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_dataset.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_dataset.csv")

# Select numerical columns
numeric_df1 = df1.select_dtypes(include="number")
numeric_df2 = df2.select_dtypes(include="number")

# Create correlation heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

corr1 = numeric_df1.corr()
corr2 = numeric_df2.corr()

import seaborn as sns

sns.heatmap(corr1, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("Credit Card Fraud Correlation")

sns.heatmap(corr2, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
axes[1].set_title("Fraud Detection Correlation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Correlation heatmaps saved successfully!")