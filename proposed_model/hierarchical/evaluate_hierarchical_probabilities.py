import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hierarchical_inference import HierarchicalPredictor

# ==========================================================
# Configuration
# ==========================================================

FEATURE_ROOT = os.path.join(PROJECT_ROOT, "features")

LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "hierarchical_final")

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# Load labels
# ==========================================================

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

class_names = list(label_encoder.classes_)

print("\n========================================")
print("7-CLASS LABEL ORDER")
print("========================================")

for i, name in enumerate(class_names):
    print(i, ":", name)


# ==========================================================
# Recreate EXACT test split
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

test_df = test_df.reset_index(drop=True)

print("\nTest samples:", len(test_df))


# ==========================================================
# Predictor
# ==========================================================

predictor = HierarchicalPredictor()


# ==========================================================
# Storage
# ==========================================================

y_true = []
y_pred = []

all_probabilities = []

binary_true = []
binary_probabilities = []


# ==========================================================
# Inference
# ==========================================================

print("\n========================================")
print("HIERARCHICAL PROBABILITY INFERENCE")
print("========================================")

for i, row in test_df.iterrows():

    filename = row["filename"]
    true_label = row["label"]

    # ------------------------------------------------------
    # Load features
    # ------------------------------------------------------

    mfcc = np.load(os.path.join(FEATURE_ROOT, "mfcc", filename))

    mel = np.load(os.path.join(FEATURE_ROOT, "mel", filename))

    chroma = np.load(os.path.join(FEATURE_ROOT, "chroma", filename))

    # ------------------------------------------------------
    # Prepare tensors
    # ------------------------------------------------------

    mfcc, mel, chroma = predictor.prepare_tensors(mfcc, mel, chroma)

    # ------------------------------------------------------
    # Stage 1: Binary model
    # ------------------------------------------------------

    with torch.no_grad():

        binary_outputs, _, _, _ = predictor.binary_model(mfcc, mel, chroma)

        binary_probs = F.softmax(binary_outputs, dim=1)[0]

    # ------------------------------------------------------
    # Binary probabilities
    #
    # IMPORTANT:
    # Binary class 0 = Normal
    # Binary class 1 = Abnormal
    # ------------------------------------------------------

    p_normal = binary_probs[0].item()
    p_abnormal = binary_probs[1].item()

    # ------------------------------------------------------
    # Ground truth binary label
    # ------------------------------------------------------

    true_binary = 0 if true_label == "Normal" else 1

    binary_true.append(true_binary)

    binary_probabilities.append(p_abnormal)

    # ------------------------------------------------------
    # Stage 2: Abnormal model
    # ------------------------------------------------------

    with torch.no_grad():

        abnormal_outputs, _, _, _ = predictor.abnormal_model(mfcc, mel, chroma)

        abnormal_probs = F.softmax(abnormal_outputs, dim=1)[0]

    # ------------------------------------------------------
    # Build final 7-class probability vector
    # ------------------------------------------------------

    final_probs = np.zeros(len(class_names), dtype=np.float64)

    for abnormal_index, abnormal_class in enumerate(predictor.abnormal_classes):

        original_index = class_names.index(abnormal_class)

        final_probs[original_index] = p_abnormal * abnormal_probs[abnormal_index].item()

    # Normal probability

    normal_index = class_names.index("Normal")

    final_probs[normal_index] = p_normal

    # ------------------------------------------------------
    # Numerical normalization
    # ------------------------------------------------------

    final_probs = final_probs / final_probs.sum()

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predicted_index = np.argmax(final_probs)

    prediction = class_names[predicted_index]

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    y_true.append(true_label)
    y_pred.append(prediction)

    all_probabilities.append(final_probs)

    if (i + 1) % 100 == 0:
        print(f"Processed " f"{i + 1}/{len(test_df)} samples")


# ==========================================================
# Convert arrays
# ==========================================================

y_true = np.array(y_true)

y_pred = np.array(y_pred)

all_probabilities = np.array(all_probabilities)

binary_true = np.array(binary_true)

binary_probabilities = np.array(binary_probabilities)


# ==========================================================
# Verify probabilities
# ==========================================================

probability_sums = all_probabilities.sum(axis=1)

print("\n========================================")
print("PROBABILITY VALIDATION")
print("========================================")

print("Minimum probability sum:", probability_sums.min())

print("Maximum probability sum:", probability_sums.max())


# ==========================================================
# Standard metrics
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)

recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)


# ==========================================================
# ROC-AUC
# ==========================================================

y_true_encoded = label_encoder.transform(y_true)

roc_auc = roc_auc_score(
    y_true_encoded, all_probabilities, multi_class="ovr", average="weighted"
)


# ==========================================================
# PR-AUC
# ==========================================================

y_true_onehot = np.zeros_like(all_probabilities)

for i, label_index in enumerate(y_true_encoded):
    y_true_onehot[i, label_index] = 1


pr_auc = average_precision_score(y_true_onehot, all_probabilities, average="weighted")


# ==========================================================
# Binary metrics
# ==========================================================

binary_predictions = (binary_probabilities >= 0.5).astype(int)

binary_accuracy = accuracy_score(binary_true, binary_predictions)

binary_macro_f1 = f1_score(binary_true, binary_predictions, average="macro")

binary_balanced_accuracy = balanced_accuracy_score(binary_true, binary_predictions)

binary_roc_auc = roc_auc_score(binary_true, binary_probabilities)

binary_pr_auc = average_precision_score(binary_true, binary_probabilities)


# ==========================================================
# Classification report
# ==========================================================

report = classification_report(y_true, y_pred, labels=class_names, zero_division=0)


# ==========================================================
# Print results
# ==========================================================

print("\n========================================")
print("HIERARCHICAL PROBABILITY EVALUATION")
print("========================================")

print(report)

print(f"Accuracy           : {accuracy:.4f}")

print(f"Precision          : {precision:.4f}")

print(f"Recall             : {recall:.4f}")

print(f"Weighted F1        : {weighted_f1:.4f}")

print(f"Macro F1           : {macro_f1:.4f}")

print(f"Balanced Accuracy  : " f"{balanced_accuracy:.4f}")

print(f"ROC AUC            : " f"{roc_auc:.4f}")

print(f"PR AUC             : " f"{pr_auc:.4f}")


# ==========================================================
# Binary results
# ==========================================================

print("\n========================================")
print("BINARY PROBABILITY METRICS")
print("========================================")

print(f"Accuracy           : " f"{binary_accuracy:.4f}")

print(f"Macro F1           : " f"{binary_macro_f1:.4f}")

print(f"Balanced Accuracy  : " f"{binary_balanced_accuracy:.4f}")

print(f"ROC AUC            : " f"{binary_roc_auc:.4f}")

print(f"PR AUC             : " f"{binary_pr_auc:.4f}")


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred, labels=class_names)

print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# Save metrics
# ==========================================================

metrics_path = os.path.join(RESULT_DIR, "probability_metrics.txt")

with open(metrics_path, "w") as f:

    f.write("HIERARCHICAL PROBABILITY EVALUATION\n")

    f.write("=" * 60 + "\n\n")

    f.write(report)

    f.write(f"\nAccuracy           : " f"{accuracy:.4f}\n")

    f.write(f"Precision          : " f"{precision:.4f}\n")

    f.write(f"Recall             : " f"{recall:.4f}\n")

    f.write(f"Weighted F1        : " f"{weighted_f1:.4f}\n")

    f.write(f"Macro F1           : " f"{macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : " f"{balanced_accuracy:.4f}\n")

    f.write(f"ROC AUC            : " f"{roc_auc:.4f}\n")

    f.write(f"PR AUC             : " f"{pr_auc:.4f}\n")

    f.write("\nBINARY STAGE\n")

    f.write(f"Accuracy           : " f"{binary_accuracy:.4f}\n")

    f.write(f"Macro F1           : " f"{binary_macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : " f"{binary_balanced_accuracy:.4f}\n")

    f.write(f"ROC AUC            : " f"{binary_roc_auc:.4f}\n")

    f.write(f"PR AUC             : " f"{binary_pr_auc:.4f}\n")


# ==========================================================
# Save probabilities
# ==========================================================

probability_df = test_df[["filename", "label"]].copy()

for i, class_name in enumerate(class_names):

    probability_df[f"prob_{class_name}"] = all_probabilities[:, i]

probability_df["prediction"] = y_pred

probability_df["confidence"] = np.max(all_probabilities, axis=1)

probability_df.to_csv(
    os.path.join(RESULT_DIR, "hierarchical_probabilities.csv"), index=False
)


# ==========================================================
# Top-3 predictions
# ==========================================================

top3_rows = []

for i in range(len(test_df)):

    indices = np.argsort(all_probabilities[i])[::-1][:3]

    row = {
        "filename": test_df.iloc[i]["filename"],
        "true_label": y_true[i],
    }

    for rank, idx in enumerate(indices, start=1):

        row[f"top{rank}_class"] = class_names[idx]

        row[f"top{rank}_prob"] = all_probabilities[i, idx]

    top3_rows.append(row)


top3_df = pd.DataFrame(top3_rows)

top3_df.to_csv(os.path.join(RESULT_DIR, "top3_predictions.csv"), index=False)


# ==========================================================
# Finished
# ==========================================================

print("\n========================================")
print("PROBABILITY EVALUATION FINISHED")
print("========================================")

print("Saved:")

print(os.path.join(RESULT_DIR, "probability_metrics.txt"))

print(os.path.join(RESULT_DIR, "hierarchical_probabilities.csv"))

print(os.path.join(RESULT_DIR, "top3_predictions.csv"))
