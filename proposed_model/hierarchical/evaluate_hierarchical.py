import os
import sys

# Allow imports from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

from multifeature_dataset import MultiFeatureDataset
from hierarchical_inference import HierarchicalPredictor

# ==========================================================
# Paths
# ==========================================================

FEATURE_ROOT = "features"
LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

RESULT_DIR = "results/hierarchical_final"

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])


print("\nOriginal Distribution:")
print(df["label"].value_counts())


# ==========================================================
# IMPORTANT:
# Recreate EXACT same test split used during training
# ==========================================================

trainval_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_encoded"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.125,
    random_state=42,
    stratify=trainval_df["label_encoded"],
)


print("\n========================================")
print("HIERARCHICAL TEST SET")
print("========================================")

print(test_df["label"].value_counts())

print("\nTest Samples:", len(test_df))


# ==========================================================
# Predictor
# ==========================================================

predictor = HierarchicalPredictor()


# ==========================================================
# Prediction
# ==========================================================

y_true = []
y_pred = []

binary_true = []
binary_pred = []


print("\n========================================")
print("RUNNING HIERARCHICAL INFERENCE")
print("========================================")


for i, row in test_df.reset_index(drop=True).iterrows():

    filename = row["filename"]

    true_label = row["label"]

    # ------------------------------------------------------
    # Load features directly
    # ------------------------------------------------------

    mfcc_path = os.path.join(
        FEATURE_ROOT,
        "mfcc",
        filename,
    )

    mel_path = os.path.join(
        FEATURE_ROOT,
        "mel",
        filename,
    )

    chroma_path = os.path.join(
        FEATURE_ROOT,
        "chroma",
        filename,
    )

    mfcc = np.load(mfcc_path)
    mel = np.load(mel_path)
    chroma = np.load(chroma_path)

    mfcc, mel, chroma = predictor.prepare_tensors(
        mfcc,
        mel,
        chroma,
    )

    # ------------------------------------------------------
    # Stage 1: Binary
    # ------------------------------------------------------

    with torch.no_grad():

        binary_outputs, _, _, _ = predictor.binary_model(
            mfcc,
            mel,
            chroma,
        )

        binary_probs = F.softmax(
            binary_outputs,
            dim=1,
        )

        binary_prediction = torch.argmax(
            binary_probs,
            dim=1,
        ).item()

    # Ground-truth binary label
    if true_label == "Normal":
        true_binary = 0
    else:
        true_binary = 1

    binary_true.append(true_binary)
    binary_pred.append(binary_prediction)

    # ------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------

    if binary_prediction == 0:

        final_prediction = "Normal"

    else:

        with torch.no_grad():

            abnormal_outputs, _, _, _ = predictor.abnormal_model(
                mfcc,
                mel,
                chroma,
            )

            abnormal_prediction = torch.argmax(
                abnormal_outputs,
                dim=1,
            ).item()

        final_prediction = predictor.abnormal_classes[abnormal_prediction]

    y_true.append(true_label)
    y_pred.append(final_prediction)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(test_df)} samples")


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred,
)


# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_true,
    y_pred,
    labels=label_encoder.classes_,
    zero_division=0,
)


print("\n========================================")
print("HIERARCHICAL TEST EVALUATION")
print("========================================")

print(report)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {balanced_acc:.4f}")


# ==========================================================
# Binary Metrics
# ==========================================================

binary_accuracy = accuracy_score(
    binary_true,
    binary_pred,
)

binary_macro_f1 = f1_score(
    binary_true,
    binary_pred,
    average="macro",
)

binary_balanced_acc = balanced_accuracy_score(
    binary_true,
    binary_pred,
)


print("\n========================================")
print("BINARY STAGE CHECK")
print("========================================")

print(f"Binary Accuracy          : " f"{binary_accuracy:.4f}")

print(f"Binary Macro F1          : " f"{binary_macro_f1:.4f}")

print(f"Binary Balanced Accuracy : " f"{binary_balanced_acc:.4f}")


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=label_encoder.classes_,
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# Save Metrics
# ==========================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "metrics.txt",
)

with open(metrics_path, "w") as f:

    f.write("HIERARCHICAL TEST EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(report)
    f.write("\n")

    f.write(f"Accuracy           : {accuracy:.4f}\n")

    f.write(f"Precision          : {precision:.4f}\n")

    f.write(f"Recall             : {recall:.4f}\n")

    f.write(f"Weighted F1        : {weighted_f1:.4f}\n")

    f.write(f"Macro F1           : {macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : {balanced_acc:.4f}\n")

    f.write("\nBINARY STAGE\n")
    f.write(f"Accuracy           : {binary_accuracy:.4f}\n")
    f.write(f"Macro F1           : {binary_macro_f1:.4f}\n")
    f.write(f"Balanced Accuracy  : {binary_balanced_acc:.4f}\n")


# ==========================================================
# Save Classification Report
# ==========================================================

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt",
    ),
    "w",
) as f:

    f.write(report)


# ==========================================================
# Plot Confusion Matrix
# ==========================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Hierarchical Model Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png",
    ),
    dpi=300,
)

plt.close()


# ==========================================================
# Normalized Confusion Matrix
# ==========================================================

cm_normalized = cm.astype(float) / cm.sum(
    axis=1,
    keepdims=True,
)

cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Hierarchical Model Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix_normalized.png",
    ),
    dpi=300,
)

plt.close()


# ==========================================================
# Save Predictions
# ==========================================================

predictions_df = test_df[["filename", "label"]].copy()

predictions_df["prediction"] = y_pred

predictions_df["correct"] = predictions_df["label"] == predictions_df["prediction"]

predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "predictions.csv",
    ),
    index=False,
)


# ==========================================================
# Finished
# ==========================================================

print("\n========================================")
print("HIERARCHICAL EVALUATION FINISHED")
print("========================================")

print("Results saved to:")
print(RESULT_DIR)

print("\nFiles:")

print(
    os.path.join(
        RESULT_DIR,
        "metrics.txt",
    )
)

print(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt",
    )
)

print(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png",
    )
)

print(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix_normalized.png",
    )
)

print(
    os.path.join(
        RESULT_DIR,
        "predictions.csv",
    )
)
