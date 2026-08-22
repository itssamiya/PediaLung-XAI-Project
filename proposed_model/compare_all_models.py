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

    if not os.path.exists(metrics_file):
        continue

    metrics = {"Model": model_name}

    # -----------------------------------------
    # Read standard metrics
    # -----------------------------------------
    with open(metrics_file, "r") as f:
        text = f.read()

    patterns = {
        "Accuracy": r"Accuracy\s*:\s*([0-9.]+)",
        "Precision": r"Precision\s*:\s*([0-9.]+)",
        "Recall": r"Recall\s*:\s*([0-9.]+)",
        "Weighted F1": r"Weighted F1\s*:\s*([0-9.]+)",
        "Macro F1": r"Macro F1\s*:\s*([0-9.]+)",
        "Balanced Accuracy": r"Balanced Accuracy\s*:\s*([0-9.]+)",
        "ROC AUC": r"ROC AUC\s*:\s*([0-9.]+)",
        "PR AUC": r"PR AUC\s*:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = None

    # -----------------------------------------
    # Hierarchical model probability metrics
    # -----------------------------------------
    if model_name == "hierarchical_final":

        probability_file = os.path.join(model_path, "probability_metrics.txt")

        if os.path.exists(probability_file):

            with open(probability_file, "r") as f:
                probability_text = f.read()

            roc_match = re.search(
                r"ROC AUC\s*:\s*([0-9.]+)", probability_text, re.IGNORECASE
            )

            pr_match = re.search(
                r"PR AUC\s*:\s*([0-9.]+)", probability_text, re.IGNORECASE
            )

            if roc_match:
                metrics["ROC AUC"] = float(roc_match.group(1))

            if pr_match:
                metrics["PR AUC"] = float(pr_match.group(1))

    models.append(metrics)


# -----------------------------------------
# Create comparison table
# -----------------------------------------
comparison_df = pd.DataFrame(models)

comparison_df = comparison_df.sort_values(by="Accuracy", ascending=False)

# Save
output_path = os.path.join(RESULTS_DIR, "model_comparison.csv")

comparison_df.to_csv(output_path, index=False)

print(comparison_df)

print("\nSaved:")
print(output_path)
