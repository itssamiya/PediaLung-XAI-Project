import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

save_dir = RESULTS_DIR

metrics = [
    ("Accuracy", "accuracy_comparison.png"),
    ("Weighted F1", "weighted_f1_comparison.png"),
    ("Macro F1", "macro_f1_comparison.png"),
    ("Balanced Accuracy", "balanced_accuracy_comparison.png"),
    ("ROC AUC", "roc_auc_comparison.png"),
    ("PR AUC", "pr_auc_comparison.png"),
]

for metric, filename in metrics:

    plt.figure(figsize=(7, 4))

    bars = plt.bar(df["Model"], df[metric])

    plt.ylabel(metric)
    plt.title(metric + " Comparison")

    plt.grid(axis="y", alpha=0.3)

    plt.ylim(0, max(df[metric]) * 1.08)

    for bar in bars:

        h = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

    plt.close()

print("All comparison figures saved.")
