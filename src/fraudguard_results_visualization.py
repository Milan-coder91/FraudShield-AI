
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

results = pd.read_csv(
    OUTPUT_DIR / "fraudguard_final_results.csv"
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

for dataset in results["Dataset"].unique():

    data = results[results["Dataset"] == dataset]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):

        axes[i].bar(
            data["Model"],
            data[metric]
        )

        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Score")
        axes[i].tick_params(
            axis="x",
            rotation=15
        )

        for bar in axes[i].patches:
            value = bar.get_height()

            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center"
            )

    fig.suptitle(
        dataset + " - Final Model Evaluation",
        fontsize=14
    )

    plt.tight_layout()

    filename = (
        dataset.lower().replace(" ", "_")
        + "_final_evaluation.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

print("\nFraudGuard Visualization Completed!")
print("Graphs saved in outputs folder.")
