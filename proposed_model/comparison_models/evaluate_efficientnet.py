import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Disable MKLDNN to avoid CPU primitive errors with EfficientNet
torch.backends.mkldnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

# ============================================================
# CPU / MEMORY SETTINGS
# ============================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORT DATASET
# ============================================================

from multifeature_dataset import MultiFeatureDataset

# ============================================================
# PATHS
# ============================================================

FEATURE_ROOT = os.path.join(
    PROJECT_ROOT,
    "features",
)

LABEL_CSV = os.path.join(
    FEATURE_ROOT,
    "labels.csv",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "comparison_models",
    "saved_models",
    "efficientnet_b0_best.pth",
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "comparison_models",
    "results",
    "efficientnet_b0",
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 1
TARGET_HEIGHT = 128
TARGET_WIDTH = 259

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# START
# ============================================================

print("=" * 60)
print("EFFICIENTNET-B0 TEST EVALUATION")
print("=" * 60)
print("Device:", DEVICE)
print("Model:", MODEL_PATH)

# ============================================================
# FEATURE CONVERSION
# ============================================================


def convert_features(mfcc, mel, chroma):
    mfcc = F.interpolate(
        mfcc,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    mel = F.interpolate(
        mel,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    chroma = F.interpolate(
        chroma,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    x = torch.cat([mfcc, mel, chroma], dim=1)
    return x


# ============================================================
# EFFICIENTNET-B0 MODEL
# ============================================================


class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# LOAD LABEL DATA
# ============================================================

print("\nLoading dataset...")
df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
num_classes = len(label_encoder.classes_)

print("\nClasses:")
for i, cls in enumerate(label_encoder.classes_):
    print(i, ":", cls)

# ============================================================
# DATA SPLIT
# ============================================================

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

print("\n==============================")
print("DATA SPLIT")
print("==============================")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

# ============================================================
# TEST DATASET & LOAD
# ============================================================

test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root=FEATURE_ROOT,
    train=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("\nLoading EfficientNet-B0...")
model = EfficientNetB0Model(num_classes=num_classes).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("EfficientNet-B0 loaded successfully.")

# ============================================================
# INFERENCE
# ============================================================

print("\n========================================")
print("RUNNING EFFICIENTNET-B0 TEST INFERENCE")
print("========================================")

y_true = []
y_pred = []
y_prob = []
processed = 0

with torch.no_grad():
    for mfcc, mel, chroma, labels in test_loader:
        mfcc = mfcc.to(DEVICE)
        mel = mel.to(DEVICE)
        chroma = chroma.to(DEVICE)
        labels = labels.to(DEVICE)

        inputs = convert_features(mfcc, mel, chroma)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

        processed += labels.size(0)
        if processed % 100 < labels.size(0):
            print(f"Processed {processed}/{len(test_dataset)}")

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ============================================================
# METRICS & CONFUSION MATRIX
# ============================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)

print("\n========================================")
print("EFFICIENTNET-B0 TEST RESULTS")
print("========================================")
print(report)
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")

# ============================================================
# SAVE METRICS & TXT REPORTS
# ============================================================

metrics_path = os.path.join(RESULT_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("EFFICIENTNET-B0 TEST EVALUATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Samples       : {len(test_dataset)}\n")
    f.write(f"Accuracy           : {accuracy:.4f}\n")
    f.write(f"Precision          : {precision:.4f}\n")
    f.write(f"Recall             : {recall:.4f}\n")
    f.write(f"Weighted F1        : {weighted_f1:.4f}\n")
    f.write(f"Macro F1           : {macro_f1:.4f}\n")
    f.write(f"Balanced Accuracy  : {balanced_accuracy:.4f}\n")

report_path = os.path.join(RESULT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

cm_path = os.path.join(RESULT_DIR, "confusion_matrix.csv")
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
cm_df.to_csv(cm_path)

predictions_path = os.path.join(RESULT_DIR, "predictions.csv")
predictions = test_df.copy()
predictions["true_class"] = label_encoder.inverse_transform(y_true)
predictions["predicted_class"] = label_encoder.inverse_transform(y_pred)
predictions["confidence"] = np.max(y_prob, axis=1)
predictions.to_csv(predictions_path, index=False)

# ============================================================
# GENERATE CONFUSION MATRIX IMAGES (RAW & NORMALIZED)
# ============================================================

# 1. Raw Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_,
)
disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)
plt.title("EfficientNet-B0 Confusion Matrix")
plt.tight_layout()
cm_img_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
plt.close()

# 2. Normalized Confusion Matrix Plot
cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
fig, ax = plt.subplots(figsize=(8, 8))
disp_norm = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=label_encoder.classes_,
)
disp_norm.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    values_format=".1%",
)
plt.title("EfficientNet-B0 Normalized Confusion Matrix (Row-wise)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_norm_img_path = os.path.join(RESULT_DIR, "confusion_matrix_normalized.png")
plt.savefig(cm_norm_img_path, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FINISHED
# ============================================================

print("\nResults saved to:")
print(metrics_path)
print(report_path)
print(cm_path)
print(predictions_path)
print(cm_img_path)
print(cm_norm_img_path)

print("\n========================================")
print("EFFICIENTNET-B0 EVALUATION FINISHED")
print("========================================")
