import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent

df1 = pd.read_csv(DATA_DIR / "credit_card_fraud_dataset.csv")
df2 = pd.read_csv(DATA_DIR / "fraud_detection_dataset.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First graph
df1["IsFraud"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], rot=0
)
axes[0].set_title("Credit Card Fraud Distribution")
axes[0].set_xlabel("Is Fraud (0 = No, 1 = Yes)")
axes[0].set_ylabel("Number of Transactions")

# Second graph
df2["Fraudulent"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], rot=0
)
axes[1].set_title("Fraud Detection Distribution")
axes[1].set_xlabel("Fraudulent (0 = No, 1 = Yes)")
axes[1].set_ylabel("Number of Transactions")

plt.tight_layout()

# Save both graphs together
plt.savefig(
    OUTPUT_DIR / "both_fraud_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Both graphs saved successfully!")