import os
import pandas as pd
import matplotlib.pyplot as plt

from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

df = pd.read_csv(os.path.join(SAVE_DIR, "experiment_comparison.csv"))


metrics = [
    "Accuracy",
    "Weighted_F1",
    "Macro_F1",
    "Balanced_Accuracy",
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(df["Experiment"], df[metric])

    plt.ylabel(metric.replace("_", " "))

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, f"{metric}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

print("Experiment comparison figures saved.")
