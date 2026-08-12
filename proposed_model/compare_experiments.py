import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

experiments = [
    "baseline",
    "residual",
    "residual_se",
    "fusion",
    "proposed",
]

rows = []

for exp in experiments:

    metrics_file = os.path.join(
        RESULTS_DIR,
        exp,
        "metrics.txt",
    )

    if not os.path.exists(metrics_file):
        print(f"{exp} -> metrics.txt not found")
        continue

    metrics = {}

    with open(metrics_file, "r") as f:

        for line in f:

            if ":" not in line:
                continue

            key, value = line.split(":")

            metrics[key.strip()] = float(value.strip())

    metrics["Experiment"] = exp

    rows.append(metrics)

df = pd.DataFrame(rows)

cols = [
    "Experiment",
    "Accuracy",
    "Precision",
    "Recall",
    "Weighted_F1",
    "Macro_F1",
    "Balanced_Accuracy",
]

df = df[cols]

save_path = os.path.join(
    RESULTS_DIR,
    "experiment_comparison.csv",
)

df.to_csv(
    save_path,
    index=False,
)

print(df)

print("\nSaved to")

print(save_path)


metrics = [
    "Accuracy",
    "Weighted_F1",
    "Macro_F1",
    "Balanced_Accuracy",
]

titles = {
    "Accuracy": "Accuracy Comparison",
    "Weighted_F1": "Weighted F1 Comparison",
    "Macro_F1": "Macro F1 Comparison",
    "Balanced_Accuracy": "Balanced Accuracy Comparison",
}

for metric in metrics:

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["Experiment"],
        df[metric],
    )

    plt.title(
        titles[metric],
        fontsize=14,
        fontweight="bold",
    )

    plt.ylabel(
        metric.replace("_", " "),
        fontsize=12,
    )

    plt.xlabel(
        "Experiment",
        fontsize=12,
    )

    plt.ylim(0, 1)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{metric}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

print("\nComparison figures saved.")
