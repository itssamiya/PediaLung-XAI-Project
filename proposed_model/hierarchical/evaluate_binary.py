import os
import sys

# Add proposed_model to Python path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader

import config as cfg

from multifeature_dataset import MultiFeatureDataset
from multibranch_model import PediaLungXAI


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("features/labels.csv")

# Normal vs Abnormal
df["binary_label"] = df["label"].apply(
    lambda x: 0 if x == "Normal" else 1
)

# Dataset split — EXACTLY the same strategy used during training
trainval_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["binary_label"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.125,
    random_state=42,
    stratify=trainval_df["binary_label"],
)

print("\n==============================")
print("BINARY TEST SET")
print("==============================")

print(test_df["binary_label"].value_counts())

print("\nNormal :", (test_df["binary_label"] == 0).sum())
print("Abnormal:", (test_df["binary_label"] == 1).sum())


# ==========================================================
# Prepare test dataframe
# ==========================================================

# MultiFeatureDataset expects label_encoded
test_df = test_df.copy()
test_df["label_encoded"] = test_df["binary_label"]


test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root="features",
    train=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# Load Binary Model
# ==========================================================

model = PediaLungXAI(
    num_classes=2,
    **cfg.MODEL_CONFIG["proposed"],
).to(device)


checkpoint = os.path.join(
    "saved_models",
    "hierarchical_binary_best.pth",
)

print("\nLoading:")
print(checkpoint)

model.load_state_dict(
    torch.load(
        checkpoint,
        map_location=device,
    )
)

model.eval()


# ==========================================================
# Prediction
# ==========================================================

true_labels = []
pred_labels = []

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

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        true_labels.extend(
            labels.numpy()
        )

        pred_labels.extend(
            predictions.cpu().numpy()
        )


true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    true_labels,
    pred_labels,
)

precision = precision_score(
    true_labels,
    pred_labels,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    true_labels,
    pred_labels,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    true_labels,
    pred_labels,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    true_labels,
    pred_labels,
    average="macro",
    zero_division=0,
)

balanced_acc = balanced_accuracy_score(
    true_labels,
    pred_labels,
)


# ==========================================================
# Classification Report
# ==========================================================

target_names = [
    "Normal",
    "Abnormal",
]

report = classification_report(
    true_labels,
    pred_labels,
    labels=[0, 1],
    target_names=target_names,
    zero_division=0,
)

print("\n========================================")
print("BINARY TEST EVALUATION")
print("========================================")

print(report)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {balanced_acc:.4f}")


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_labels,
    pred_labels,
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# Save Results
# ==========================================================

save_dir = os.path.join(
    "results",
    "hierarchical_binary",
)

os.makedirs(
    save_dir,
    exist_ok=True,
)

with open(
    os.path.join(
        save_dir,
        "test_metrics.txt",
    ),
    "w",
) as f:

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.4f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: {balanced_acc:.4f}\n"
    )


with open(
    os.path.join(
        save_dir,
        "classification_report.txt",
    ),
    "w",
) as f:

    f.write(report)


np.savetxt(
    os.path.join(
        save_dir,
        "confusion_matrix.txt",
    ),
    cm,
    fmt="%d",
)


print("\nResults saved to:")
print(save_dir)