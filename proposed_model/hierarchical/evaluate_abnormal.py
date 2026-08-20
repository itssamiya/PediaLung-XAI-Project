import os
import sys

# ==========================================================
# Add parent project directory to Python path
# ==========================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

import config as cfg

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

# ==========================================================
# Paths
# ==========================================================

FEATURE_ROOT = "features"
LABEL_CSV = os.path.join(
    FEATURE_ROOT,
    "labels.csv",
)

MODEL_PATH = os.path.join(
    "saved_models",
    "hierarchical_abnormal_best.pth",
)

RESULT_DIR = os.path.join(
    "results",
    "hierarchical_abnormal",
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(LABEL_CSV)

print("\nOriginal Distribution:")
print(df["label"].value_counts())


# ==========================================================
# Keep abnormal classes only
# ==========================================================

abnormal_classes = [
    "Fine Crackle",
    "Wheeze",
    "Coarse Crackle",
    "Rhonchi",
    "Wheeze+Crackle",
    "Stridor",
]

abnormal_df = df[df["label"].isin(abnormal_classes)].copy()

print("\nAbnormal Distribution:")
print(abnormal_df["label"].value_counts())


# ==========================================================
# Label Encoder
# ==========================================================

label_encoder = LabelEncoder()

abnormal_df["label_encoded"] = label_encoder.fit_transform(abnormal_df["label"])

print("\nAbnormal Classes:")

for i, cls in enumerate(label_encoder.classes_):
    print(i, ":", cls)


# ==========================================================
# SAME TEST SPLIT USED DURING TRAINING
# ==========================================================

from sklearn.model_selection import train_test_split

trainval_df, test_df = train_test_split(
    abnormal_df,
    test_size=0.20,
    random_state=42,
    stratify=abnormal_df["label_encoded"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.125,
    random_state=42,
    stratify=trainval_df["label_encoded"],
)


print("\n==============================")
print("TEST DISTRIBUTION")
print("==============================")

print(test_df["label"].value_counts())


# ==========================================================
# Test Dataset
# ==========================================================

test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root=FEATURE_ROOT,
    train=False,
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# Load Model
# ==========================================================

print("\n==============================")
print("LOADING ABNORMAL MODEL")
print("==============================")

model = PediaLungXAI(
    num_classes=len(label_encoder.classes_),
    **cfg.MODEL_CONFIG["proposed"],
).to(device)


print("Model Path:")
print(MODEL_PATH)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()

print("Model loaded successfully.")


# ==========================================================
# Prediction
# ==========================================================

all_labels = []
all_predictions = []
all_probabilities = []


print("\nRunning abnormal test evaluation...")


with torch.no_grad():

    for mfcc, mel, chroma, labels in test_loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        outputs, _, _, _ = model(
            mfcc,
            mel,
            chroma,
        )

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_labels.extend(labels.numpy())

        all_predictions.extend(predictions.cpu().numpy())

        all_probabilities.extend(probabilities.cpu().numpy())


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# ==========================================================
# Basic Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

balanced_acc = balanced_accuracy_score(
    all_labels,
    all_predictions,
)


# ==========================================================
# ROC-AUC
# ==========================================================

try:

    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities,
        multi_class="ovr",
        average="macro",
    )

except ValueError:

    roc_auc = None


# ==========================================================
# PR-AUC
# ==========================================================

try:

    # One-vs-rest binary indicators
    y_true_onehot = np.eye(len(label_encoder.classes_))[all_labels]

    pr_auc = average_precision_score(
        y_true_onehot,
        all_probabilities,
        average="macro",
    )

except ValueError:

    pr_auc = None


# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0,
)


print("\n========================================")
print("ABNORMAL TEST EVALUATION")
print("========================================")

print(report)

print(f"Accuracy           : {accuracy:.4f}")

print(f"Precision          : {precision:.4f}")

print(f"Recall             : {recall:.4f}")

print(f"Weighted F1        : {weighted_f1:.4f}")

print(f"Macro F1           : {macro_f1:.4f}")

print(f"Balanced Accuracy  : {balanced_acc:.4f}")

if roc_auc is not None:

    print(f"ROC AUC            : {roc_auc:.4f}")

else:

    print("ROC AUC            : N/A")


if pr_auc is not None:

    print(f"PR AUC             : {pr_auc:.4f}")

else:

    print("PR AUC             : N/A")


# ==========================================================
# Save Metrics
# ==========================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "metrics.txt",
)

with open(
    metrics_path,
    "w",
) as f:

    f.write("ABNORMAL TEST EVALUATION\n")

    f.write("========================\n\n")

    f.write(f"Accuracy           : {accuracy:.4f}\n")

    f.write(f"Precision          : {precision:.4f}\n")

    f.write(f"Recall             : {recall:.4f}\n")

    f.write(f"Weighted F1        : {weighted_f1:.4f}\n")

    f.write(f"Macro F1           : {macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : {balanced_acc:.4f}\n")

    if roc_auc is not None:

        f.write(f"ROC AUC            : {roc_auc:.4f}\n")

    if pr_auc is not None:

        f.write(f"PR AUC             : {pr_auc:.4f}\n")


# ==========================================================
# Save Classification Report
# ==========================================================

report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt",
)

with open(
    report_path,
    "w",
) as f:

    f.write(report)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=np.arange(len(label_encoder.classes_)),
)


print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# Save Confusion Matrix Figure
# ==========================================================

plt.figure(figsize=(9, 7))

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title("Abnormal Respiratory Sound Classification - Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(label_encoder.classes_))

plt.xticks(
    tick_marks,
    label_encoder.classes_,
    rotation=45,
    ha="right",
)

plt.yticks(
    tick_marks,
    label_encoder.classes_,
)

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )


plt.ylabel("True Label")

plt.xlabel("Predicted Label")

plt.tight_layout()

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png",
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==========================================================
# Normalized Confusion Matrix
# ==========================================================

cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

cm_normalized = np.nan_to_num(cm_normalized)


plt.figure(figsize=(9, 7))

plt.imshow(
    cm_normalized,
    interpolation="nearest",
)

plt.title("Abnormal Classification - Normalized Confusion Matrix")

plt.colorbar()

plt.xticks(
    tick_marks,
    label_encoder.classes_,
    rotation=45,
    ha="right",
)

plt.yticks(
    tick_marks,
    label_encoder.classes_,
)


for i in range(cm_normalized.shape[0]):

    for j in range(cm_normalized.shape[1]):

        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            horizontalalignment="center",
            color="white" if cm_normalized[i, j] > 0.5 else "black",
        )


plt.ylabel("True Label")

plt.xlabel("Predicted Label")

plt.tight_layout()

normalized_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix_normalized.png",
)

plt.savefig(
    normalized_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==========================================================
# Final
# ==========================================================

print("\nResults saved to:")

print(RESULT_DIR)

print("\nFiles:")

print(metrics_path)

print(report_path)

print(cm_path)

print(normalized_path)
