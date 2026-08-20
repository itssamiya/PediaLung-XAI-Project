import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn


import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader, WeightedRandomSampler

from multifeature_dataset import MultiFeatureDataset
from multibranch_model import PediaLungXAI
from focal_loss import FocalLoss
from utils.early_stopping import EarlyStopping

# ==========================================================
# SETTINGS
# ==========================================================

EXPERIMENT_NAME = "hierarchical_abnormal_ce_sampler"

MODEL_DIR = "saved_models"
RESULT_DIR = os.path.join("results", EXPERIMENT_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("features/labels.csv")

print("\nOriginal Distribution:")
print(df["label"].value_counts())


# ==========================================================
# SELECT ABNORMAL CLASSES
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
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

abnormal_df["label_encoded"] = label_encoder.fit_transform(abnormal_df["label"])

print("\nAbnormal Classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(i, ":", cls)


# ==========================================================
# DATA SPLIT
# ==========================================================

trainval_df, test_df = train_test_split(
    abnormal_df,
    test_size=0.10,
    random_state=42,
    stratify=abnormal_df["label_encoded"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=1 / 9,
    random_state=42,
    stratify=trainval_df["label_encoded"],
)


print("\n==============================")
print("ABNORMAL DATA SPLIT")
print("==============================")

print("\nTRAIN")
print(train_df["label"].value_counts())

print("\nVALIDATION")
print(val_df["label"].value_counts())

print("\nTEST")
print(test_df["label"].value_counts())


# ==========================================================
# DATASETS
# ==========================================================

train_dataset = MultiFeatureDataset(
    dataframe=train_df,
    feature_root="features",
    train=True,
)

val_dataset = MultiFeatureDataset(
    dataframe=val_df,
    feature_root="features",
    train=False,
)

test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root="features",
    train=False,
)


# ==========================================================
# WEIGHTED RANDOM SAMPLER
# ==========================================================

class_counts = train_df["label_encoded"].value_counts().sort_index()

print("\nClass Counts:")
print(class_counts)


class_weights = 1.0 / class_counts

sample_weights = train_df["label_encoded"].map(class_weights)

sample_weights = torch.DoubleTensor(sample_weights.values)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

print("\nWeightedRandomSampler: ENABLED")


# ==========================================================
# DATA LOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    shuffle=False,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# MODEL
# ==========================================================

model = PediaLungXAI(
    num_classes=len(label_encoder.classes_),
    **cfg.MODEL_CONFIG["proposed"],
).to(device)


print("\n==============================")
print("ABNORMAL MODEL")
print("==============================")

print("Architecture :", "proposed")
print("Classes      :", len(label_encoder.classes_))
# print("Loss         :", "Focal Loss")
# print("Gamma        :", cfg.FOCAL_GAMMA)
print("Loss         :", "Cross Entropy")
print("Sampler      :", "WeightedRandomSampler")


# ==========================================================
# LOSS
# ==========================================================

criterion = nn.CrossEntropyLoss()


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=cfg.LEARNING_RATE,
    weight_decay=cfg.WEIGHT_DECAY,
)


# ==========================================================
# SCHEDULER
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
)


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stopping = EarlyStopping(patience=10)


# ==========================================================
# METRIC FUNCTIONS
# ==========================================================


def compute_macro_f1(
    y_true,
    y_pred,
    num_classes,
):

    f1_scores = []

    for cls in range(num_classes):

        tp = np.sum((y_true == cls) & (y_pred == cls))

        fp = np.sum((y_true != cls) & (y_pred == cls))

        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        f1_scores.append(f1)

    return np.mean(f1_scores)


def compute_weighted_f1(
    y_true,
    y_pred,
    num_classes,
):

    total = len(y_true)
    weighted = 0.0

    for cls in range(num_classes):

        tp = np.sum((y_true == cls) & (y_pred == cls))

        fp = np.sum((y_true != cls) & (y_pred == cls))

        fn = np.sum((y_true == cls) & (y_pred != cls))

        support = np.sum(y_true == cls)

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        weighted += f1 * support

    return weighted / total


# ==========================================================
# TRAINING
# ==========================================================

best_macro_f1 = 0.0

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "macro_f1": [],
    "weighted_f1": [],
    "balanced_accuracy": [],
}


for epoch in range(40):

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for mfcc, mel, chroma, labels in train_loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _, _, _ = model(
            mfcc,
            mel,
            chroma,
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1,
        )

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100.0 * correct / total

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for mfcc, mel, chroma, labels in val_loader:

            mfcc = mfcc.to(device)
            mel = mel.to(device)
            chroma = chroma.to(device)
            labels = labels.to(device)

            outputs, _, _, _ = model(
                mfcc,
                mel,
                chroma,
            )

            loss = criterion(
                outputs,
                labels,
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1,
            )

            all_labels.extend(labels.cpu().numpy())

            all_preds.extend(predicted.cpu().numpy())

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)

    val_accuracy = 100.0 * val_correct / val_total

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    macro_f1 = compute_macro_f1(
        all_labels,
        all_preds,
        len(label_encoder.classes_),
    )

    weighted_f1 = compute_weighted_f1(
        all_labels,
        all_preds,
        len(label_encoder.classes_),
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_preds,
    )

    # ------------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------------

    scheduler.step(macro_f1)

    # ------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR,
                "hierarchical_abnormal_ce_sampler_best.pth",
            ),
        )

        print(f"\nBest model saved: " f"Macro F1 = {macro_f1:.4f}")

    # ------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------

    history["epoch"].append(epoch + 1)

    history["train_loss"].append(train_loss)

    history["val_loss"].append(val_loss)

    history["train_acc"].append(train_accuracy)

    history["val_acc"].append(val_accuracy)

    history["macro_f1"].append(macro_f1)

    history["weighted_f1"].append(weighted_f1)

    history["balanced_accuracy"].append(balanced_acc)

    print(
        f"Epoch [{epoch+1}/40] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Balanced Acc: {balanced_acc:.4f}"
    )

    # ------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------

    if early_stopping.step(macro_f1):

        print("\nEarly stopping triggered.")

        break


# ==========================================================
# SAVE HISTORY
# ==========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "history.csv",
    ),
    index=False,
)


print("\n========================================")
print("ABNORMAL TRAINING FINISHED")
print("========================================")

print(f"Best Validation Macro F1: " f"{best_macro_f1:.4f}")

print("Model saved to:")

print(
    os.path.join(
        MODEL_DIR,
        "hierarchical_abnormal_ce_sampler_best.pth",
    )
)

print("History saved to:")

print(
    os.path.join(
        RESULT_DIR,
        "history.csv",
    )
)
