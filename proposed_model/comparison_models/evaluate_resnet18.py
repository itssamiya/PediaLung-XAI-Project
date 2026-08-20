import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from torch.utils.data import DataLoader
import torch.nn.functional as F

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
# IMPORTS FROM TRAINING PIPELINE
# ============================================================

from multifeature_dataset import MultiFeatureDataset

# Import the model directly from the training file
from comparison_models.train_resnet18 import ResNet18, convert_features

# ============================================================
# PATHS
# ============================================================

FEATURE_ROOT = os.path.join(PROJECT_ROOT, "features")

LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "comparison_models", "saved_models", "resnet18_best.pth"
)

RESULT_DIR = os.path.join(PROJECT_ROOT, "comparison_models", "results", "resnet18")

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# START
# ============================================================

print("=" * 60)
print("RESNET-18 TEST EVALUATION")
print("=" * 60)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)


# ============================================================
# LOAD LABEL DATA
# ============================================================

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

num_classes = len(label_encoder.classes_)

print("\nClasses:")

for i, cls in enumerate(label_encoder.classes_):
    print(i, ":", cls)


# ============================================================
# EXACT SAME DATA SPLIT AS TRAINING
# ============================================================

from sklearn.model_selection import train_test_split

trainval_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label_encoded"]
)

train_df, val_df = train_test_split(
    trainval_df, test_size=0.125, random_state=42, stratify=trainval_df["label_encoded"]
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = MultiFeatureDataset(
    dataframe=test_df, feature_root=FEATURE_ROOT, train=False
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ============================================================
# LOAD RESNET-18
# ============================================================

print("\nLoading ResNet-18...")

model = ResNet18(num_classes=num_classes).to(DEVICE)


# Training saved raw state_dict:
# torch.save(model.state_dict(), MODEL_PATH)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(state_dict)

model.eval()

print("ResNet-18 loaded successfully.")


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nTotal Parameters     :", f"{total_params:,}")
print("Trainable Parameters :", f"{trainable_params:,}")


# ============================================================
# INFERENCE
# ============================================================

print("\n========================================")
print("RUNNING RESNET-18 TEST INFERENCE")
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

        # EXACT SAME CONVERSION USED DURING TRAINING
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
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)

recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("RESNET-18 TEST RESULTS")
print("========================================")

print(report)

print(f"Accuracy           : {accuracy:.4f}")

print(f"Precision          : {precision:.4f}")

print(f"Recall             : {recall:.4f}")

print(f"Weighted F1        : {weighted_f1:.4f}")

print(f"Macro F1           : {macro_f1:.4f}")

print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(RESULT_DIR, "metrics.txt")

with open(metrics_path, "w") as f:

    f.write("RESNET-18 TEST EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Test Samples       : {len(test_dataset)}\n")

    f.write(f"Total Parameters   : {total_params:,}\n")

    f.write(f"Accuracy           : {accuracy:.4f}\n")

    f.write(f"Precision          : {precision:.4f}\n")

    f.write(f"Recall             : {recall:.4f}\n")

    f.write(f"Weighted F1        : {weighted_f1:.4f}\n")

    f.write(f"Macro F1           : {macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : {balanced_accuracy:.4f}\n")


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(RESULT_DIR, "classification_report.txt")

with open(report_path, "w") as f:
    f.write(report)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(RESULT_DIR, "confusion_matrix.csv")

cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

cm_df.to_csv(cm_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test_df.copy()

predictions["true_class"] = label_encoder.inverse_transform(y_true)

predictions["predicted_class"] = label_encoder.inverse_transform(y_pred)

predictions["confidence"] = np.max(y_prob, axis=1)

predictions_path = os.path.join(RESULT_DIR, "predictions.csv")

predictions.to_csv(predictions_path, index=False)


# ============================================================
# FINISHED
# ============================================================

print("\nResults saved to:")

print(metrics_path)
print(report_path)
print(cm_path)
print(predictions_path)

print("\n========================================")
print("RESNET-18 EVALUATION FINISHED")
print("========================================")
