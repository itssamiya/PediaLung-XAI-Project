import os
import re
import pandas as pd

RESULTS_DIR = "results"

models = []

for model_name in sorted(os.listdir(RESULTS_DIR)):

    model_path = os.path.join(RESULTS_DIR, model_name)

    if not os.path.isdir(model_path):
        continue

    metrics_file = os.path.join(model_path, "metrics.txt")

    report_file = os.path.join(model_path, "classification_report.txt")

    if not os.path.exists(metrics_file):
        continue

    metrics = {"Model": model_name}

    with open(metrics_file, "r") as f:

        text = f.read()

    patterns = {
        "Accuracy": r"Accuracy:\s*([0-9.]+)",
        "Precision": r"Precision:\s*([0-9.]+)",
        "Recall": r"Recall:\s*([0-9.]+)",
        "Weighted F1": r"Weighted F1:\s*([0-9.]+)",
        "Macro F1": r"Macro F1:\s*([0-9.]+)",
        "Balanced Accuracy": r"Balanced Accuracy:\s*([0-9.]+)",
        "ROC AUC": r"ROC AUC:\s*([0-9.]+)",
        "PR AUC": r"PR AUC:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = None

    models.append(metrics)

comparison_df = pd.DataFrame(models)

comparison_df = comparison_df.sort_values(by="Accuracy", ascending=False)

comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

print(comparison_df)

print("\nSaved:")
print(os.path.join(RESULTS_DIR, "model_comparison.csv"))
