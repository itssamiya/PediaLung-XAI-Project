import os
import re
import pandas as pd

# ==========================================================
# PROJECT MODEL COMPARISON
# ==========================================================
# Compares exactly three models:
#
# 1. PediaLung-XAI
# 2. ResNet-18
# 3. EfficientNet-B0
#
# This file is ONLY for the project comparison.
# It is separate from compare_all_models.py, which is used
# for the research/paper experiments.
# ==========================================================


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROPOSED_RESULTS = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "results",
    "proposed_focal_sampler",
)

COMPARISON_RESULTS = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "comparison_models",
    "results",
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "results",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# MODEL RESULT FILES
# ==========================================================

MODEL_FILES = {
    "PediaLung-XAI": os.path.join(
        PROPOSED_RESULTS,
        "metrics.txt",
    ),
    "ResNet-18": os.path.join(
        COMPARISON_RESULTS,
        "resnet18",
        "metrics.txt",
    ),
    "EfficientNet-B0": os.path.join(
        COMPARISON_RESULTS,
        "efficientnet_b0",
        "metrics.txt",
    ),
}


# ==========================================================
# METRIC PATTERNS
# ==========================================================

PATTERNS = {
    "Accuracy": r"Accuracy\s*:\s*([0-9.]+)",
    "Precision": r"Precision\s*:\s*([0-9.]+)",
    "Recall": r"Recall\s*:\s*([0-9.]+)",
    "Weighted F1": r"Weighted F1\s*:\s*([0-9.]+)",
    "Macro F1": r"Macro F1\s*:\s*([0-9.]+)",
    "Balanced Accuracy": r"Balanced Accuracy\s*:\s*([0-9.]+)",
    "ROC AUC": r"ROC AUC\s*:\s*([0-9.]+)",
    "PR AUC": r"PR AUC\s*:\s*([0-9.]+)",
    "Total Parameters": r"Total Parameters\s*:\s*([0-9,]+)",
    "Test Samples": r"Test Samples\s*:\s*([0-9]+)",
}


# ==========================================================
# READ METRICS
# ==========================================================


def read_metrics(model_name, metrics_file):

    metrics = {"Model": model_name}

    if not os.path.exists(metrics_file):

        print(f"\nWARNING: Metrics file not found for " f"{model_name}:")

        print(metrics_file)

        for key in PATTERNS:
            metrics[key] = None

        return metrics

    with open(
        metrics_file,
        "r",
        encoding="utf-8",
    ) as f:

        text = f.read()

    for key, pattern in PATTERNS.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            value = match.group(1)

            if key == "Total Parameters":

                value = value.replace(",", "")

                metrics[key] = int(value)

            else:

                metrics[key] = float(value)

        else:

            metrics[key] = None

    return metrics


# ==========================================================
# LOAD ALL MODELS
# ==========================================================

results = []


print("=" * 70)
print("PEDIALUNG-XAI PROJECT MODEL COMPARISON")
print("=" * 70)


for model_name, metrics_file in MODEL_FILES.items():

    print(f"\nLoading {model_name}...")

    metrics = read_metrics(
        model_name,
        metrics_file,
    )

    results.append(metrics)


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

comparison_df = pd.DataFrame(results)


# ==========================================================
# ORDER MODELS
# ==========================================================

model_order = [
    "PediaLung-XAI",
    "ResNet-18",
    "EfficientNet-B0",
]


comparison_df["Model"] = pd.Categorical(
    comparison_df["Model"],
    categories=model_order,
    ordered=True,
)


comparison_df = comparison_df.sort_values("Model")


# ==========================================================
# SAVE FULL COMPARISON
# ==========================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "project_model_comparison.csv",
)


comparison_df.to_csv(
    output_csv,
    index=False,
)


# ==========================================================
# DISPLAY
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison_df.to_string(index=False))


# ==========================================================
# DISPLAY IMPORTANT METRICS
# ==========================================================

print("\n")
print("=" * 70)
print("KEY PERFORMANCE METRICS")
print("=" * 70)


display_columns = [
    "Model",
    "Accuracy",
    "Weighted F1",
    "Macro F1",
    "Balanced Accuracy",
]


print(comparison_df[display_columns].to_string(index=False))


# ==========================================================
# BEST MODEL BY ACCURACY
# ==========================================================

valid_accuracy = comparison_df.dropna(subset=["Accuracy"])


if not valid_accuracy.empty:

    best_accuracy_model = valid_accuracy.loc[valid_accuracy["Accuracy"].idxmax()]

    print("\n")
    print("Best Accuracy:")

    print(f"{best_accuracy_model['Model']} " f"({best_accuracy_model['Accuracy']:.4f})")


# ==========================================================
# BEST MODEL BY WEIGHTED F1
# ==========================================================

valid_f1 = comparison_df.dropna(subset=["Weighted F1"])


if not valid_f1.empty:

    best_f1_model = valid_f1.loc[valid_f1["Weighted F1"].idxmax()]

    print("\n")
    print("Best Weighted F1:")

    print(f"{best_f1_model['Model']} " f"({best_f1_model['Weighted F1']:.4f})")


# ==========================================================
# BEST MODEL BY MACRO F1
# ==========================================================

valid_macro = comparison_df.dropna(subset=["Macro F1"])


if not valid_macro.empty:

    best_macro_model = valid_macro.loc[valid_macro["Macro F1"].idxmax()]

    print("\n")
    print("Best Macro F1:")

    print(f"{best_macro_model['Model']} " f"({best_macro_model['Macro F1']:.4f})")


# ==========================================================
# FINISHED
# ==========================================================

print("\n")
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("\nSaved to:")

print(output_csv)
