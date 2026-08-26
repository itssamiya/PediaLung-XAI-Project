import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)

from torch.utils.data import DataLoader

from multifeature_dataset import MultiFeatureDataset
from multibranch_model import PediaLungXAI

from config import (
    MODEL_CONFIG,
    MODEL_DIR,
    SAVE_DIR,
    EXPERIMENT_NAME,
)

# RESULT_EXPERIMENT = "proposed_focal_g2"
RESULT_EXPERIMENT = "proposed_focal_sampler"
CHECKPOINT_NAME = "proposed_focal_g2_best.pth"
SAVE_DIR = os.path.join(cfg.RESULTS_ROOT, RESULT_EXPERIMENT)

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

os.makedirs(SAVE_DIR, exist_ok=True)

#########################################################
# Load Dataset
#########################################################

df = pd.read_csv("features/labels.csv")

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_encoded"],
)

test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root="features",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
)

#########################################################
# Load Model
#########################################################

model = PediaLungXAI(
    num_classes=len(label_encoder.classes_),
    **MODEL_CONFIG[cfg.MODEL_NAME],
).to(device)

CHECKPOINT_NAME = "proposed_focal_g2_best.pth"

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, CHECKPOINT_NAME),
        map_location=device,
    )
)

model.eval()

print("\n========================================")
print("FINAL MODEL EVALUATION")
print("========================================")
print("Architecture :", cfg.MODEL_NAME)
print("Loss         : Focal Loss")
print("Gamma        :", cfg.FOCAL_GAMMA)
print("Checkpoint   :", CHECKPOINT_NAME)
print("========================================\n")

#########################################################
# Prediction
#########################################################

true_labels = []
pred_labels = []
probabilities = []

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

        probs = torch.softmax(outputs, dim=1)

        probabilities.extend(probs.cpu().numpy())

        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())

        pred_labels.extend(predicted.cpu().numpy())

probabilities = np.array(probabilities)

#########################################################
# Metrics
#########################################################

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

#########################################################
# ROC & PR AUC
#########################################################

true_binary = label_binarize(
    true_labels,
    classes=np.arange(len(label_encoder.classes_)),
)

roc_auc_list = []
pr_auc_list = []

for i in range(len(label_encoder.classes_)):

    fpr, tpr, _ = roc_curve(
        true_binary[:, i],
        probabilities[:, i],
    )

    roc_auc_list.append(auc(fpr, tpr))

    precision_curve, recall_curve, _ = precision_recall_curve(
        true_binary[:, i],
        probabilities[:, i],
    )

    pr_auc_list.append(auc(recall_curve, precision_curve))

roc_auc = np.mean(roc_auc_list)
pr_auc = np.mean(pr_auc_list)

#########################################################
# Classification Report
#########################################################

report = classification_report(
    true_labels,
    pred_labels,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0,
)

print(report)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {balanced_acc:.4f}")
print(f"ROC AUC            : {roc_auc:.4f}")
print(f"PR AUC             : {pr_auc:.4f}")

#########################################################
# Save Metrics
#########################################################

with open(
    os.path.join(
        SAVE_DIR,
        "metrics.txt",
    ),
    "w",
) as f:

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"PR AUC: {pr_auc:.4f}\n")

with open(
    os.path.join(
        SAVE_DIR,
        "classification_report.txt",
    ),
    "w",
) as f:

    f.write(report)

#########################################################
# Confusion Matrix
#########################################################

cm = confusion_matrix(
    true_labels,
    pred_labels,
)

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

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

#########################################################
# Normalized Confusion Matrix
#########################################################

cm_norm = confusion_matrix(
    true_labels,
    pred_labels,
    normalize="true",
)

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_norm,
    display_labels=label_encoder.classes_,
)

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    values_format=".1%",
)

plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confusion_matrix_normalized.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

#########################################################
# ROC Curves
#########################################################

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(label_encoder.classes_):

    fpr, tpr, _ = roc_curve(
        true_binary[:, i],
        probabilities[:, i],
    )

    class_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} (AUC={class_auc:.2f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1,
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")

plt.legend(
    fontsize=8,
    loc="lower right",
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "roc_curve.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

#########################################################
# Precision–Recall Curves
#########################################################

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(label_encoder.classes_):

    precision_curve, recall_curve, _ = precision_recall_curve(
        true_binary[:, i],
        probabilities[:, i],
    )

    class_pr_auc = auc(
        recall_curve,
        precision_curve,
    )

    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"{class_name} (AUC={class_pr_auc:.2f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")

plt.legend(
    fontsize=8,
    loc="lower left",
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "precision_recall_curve.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

#########################################################
# Finished
#########################################################

print("\n========================================")
print("Evaluation Completed Successfully")
print("========================================")

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Balanced Accuracy  : {balanced_acc:.4f}")
print(f"ROC AUC            : {roc_auc:.4f}")
print(f"PR AUC             : {pr_auc:.4f}")

print("\nSaved files:")

print(os.path.join(SAVE_DIR, "metrics.txt"))
print(os.path.join(SAVE_DIR, "classification_report.txt"))
print(os.path.join(SAVE_DIR, "confusion_matrix.png"))
print(os.path.join(SAVE_DIR, "confusion_matrix_normalized.png"))
print(os.path.join(SAVE_DIR, "roc_curve.png"))
print(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
