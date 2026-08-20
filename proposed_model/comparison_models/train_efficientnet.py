import os
import sys

# ==========================================================
# Reduce CPU / memory pressure
# ==========================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
)

from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision.models import efficientnet_b0

from multifeature_dataset import MultiFeatureDataset
from focal_loss import FocalLoss

# ==========================================================
# Configuration
# ==========================================================

FEATURE_ROOT = os.path.join(
    PROJECT_ROOT,
    "features",
)

LABEL_CSV = os.path.join(
    FEATURE_ROOT,
    "labels.csv",
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Separate result directory
RESULT_DIR = os.path.join(
    CURRENT_DIR,
    "results",
    "efficientnet_b0",
)

# Model directory
MODEL_DIR = os.path.join(
    CURRENT_DIR,
    "saved_models",
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_best.pth",
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "history.csv",
)

CLASS_NAMES_PATH = os.path.join(
    RESULT_DIR,
    "class_names.json",
)


# ==========================================================
# Training parameters
# ==========================================================

BATCH_SIZE = 8

NUM_EPOCHS = 40

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

FOCAL_GAMMA = 2.0

EARLY_STOPPING_PATIENCE = 10

TARGET_HEIGHT = 128

TARGET_WIDTH = 259


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 60)
print("EFFICIENTNET-B0 COMPARISON MODEL")
print("=" * 60)

print("Device:", device)

print("Feature root:", FEATURE_ROOT)

print("Result directory:", RESULT_DIR)


# ==========================================================
# Feature conversion
# ==========================================================


def convert_features(
    mfcc,
    mel,
    chroma,
):

    # ------------------------------------------------------
    # MFCC
    # [B, 1, 40, 94]
    # ->
    # [B, 1, 128, 259]
    # ------------------------------------------------------

    mfcc = F.interpolate(
        mfcc,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # ------------------------------------------------------
    # Mel
    # Already [B, 1, 128, 259]
    # ------------------------------------------------------

    mel = F.interpolate(
        mel,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # ------------------------------------------------------
    # Chroma
    # [B, 1, 12, 259]
    # ->
    # [B, 1, 128, 259]
    # ------------------------------------------------------

    chroma = F.interpolate(
        chroma,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # ------------------------------------------------------
    # Combine the three feature types as channels
    #
    # [B, 1, H, W] × 3
    # ->
    # [B, 3, H, W]
    # ------------------------------------------------------

    x = torch.cat(
        [
            mfcc,
            mel,
            chroma,
        ],
        dim=1,
    )

    return x


# ==========================================================
# EfficientNet-B0
# ==========================================================


class EfficientNetB0Model(nn.Module):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()

        # --------------------------------------------------
        # EfficientNet-B0
        #
        # weights=None keeps the experiment comparable
        # with the from-scratch ResNet experiment.
        # --------------------------------------------------

        self.backbone = efficientnet_b0(
            weights=None,
        )

        # --------------------------------------------------
        # EfficientNet-B0 already accepts 3 channels.
        #
        # Our three channels are:
        #
        # Channel 0 = MFCC
        # Channel 1 = Mel
        # Channel 2 = Chroma
        # --------------------------------------------------

        feature_dim = self.backbone.classifier[-1].in_features

        # --------------------------------------------------
        # Replace original ImageNet classifier
        # --------------------------------------------------

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(
                p=0.2,
            ),
            nn.Linear(
                feature_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.backbone(x)


# ==========================================================
# Metrics
# ==========================================================


def compute_metrics(
    y_true,
    y_pred,
):

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    return (
        macro_f1,
        weighted_f1,
        balanced_acc,
    )


# ==========================================================
# Load dataset
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    LABEL_CSV,
)

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

num_classes = len(label_encoder.classes_)


print("\nClasses:")

for i, cls in enumerate(label_encoder.classes_):

    print(
        i,
        ":",
        cls,
    )


# ==========================================================
# Save class names
# ==========================================================

with open(
    CLASS_NAMES_PATH,
    "w",
) as f:

    json.dump(
        label_encoder.classes_.tolist(),
        f,
        indent=4,
    )


# ==========================================================
# SAME DATA SPLIT AS RESNET
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


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print(
    "Training   :",
    len(train_df),
)

print(
    "Validation :",
    len(val_df),
)

print(
    "Testing    :",
    len(test_df),
)


# ==========================================================
# Datasets
# ==========================================================

train_dataset = MultiFeatureDataset(
    dataframe=train_df,
    feature_root=FEATURE_ROOT,
    train=True,
)


val_dataset = MultiFeatureDataset(
    dataframe=val_df,
    feature_root=FEATURE_ROOT,
    train=False,
)


test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root=FEATURE_ROOT,
    train=False,
)


# ==========================================================
# Weighted Random Sampler
# ==========================================================

class_counts = train_df["label_encoded"].value_counts().sort_index()


class_weights = 1.0 / class_counts


sample_weights = train_df["label_encoded"].map(class_weights)


sample_weights = torch.DoubleTensor(
    sample_weights.values,
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


print("\nWeightedRandomSampler: ENABLED")


# ==========================================================
# Data loaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# Model
# ==========================================================

model = EfficientNetB0Model(
    num_classes=num_classes,
).to(device)


print("\n==============================")
print("MODEL")
print("==============================")

print(model)


# ==========================================================
# Parameter count
# ==========================================================

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


print(
    "\nTotal Parameters:",
    f"{total_params:,}",
)

print(
    "Trainable Parameters:",
    f"{trainable_params:,}",
)


# ==========================================================
# Loss
# ==========================================================

criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    alpha=None,
)


# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ==========================================================
# Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
)


# ==========================================================
# Training
# ==========================================================

best_macro_f1 = 0.0

epochs_without_improvement = 0

history = []


for epoch in range(NUM_EPOCHS):

    # ======================================================
    # Training
    # ======================================================

    model.train()

    running_loss = 0.0

    train_correct = 0

    train_total = 0

    for (
        mfcc,
        mel,
        chroma,
        labels,
    ) in train_loader:

        mfcc = mfcc.to(device)

        mel = mel.to(device)

        chroma = chroma.to(device)

        labels = labels.to(device)

        # --------------------------------------------------
        # Convert to 3-channel representation
        # --------------------------------------------------

        inputs = convert_features(
            mfcc,
            mel,
            chroma,
        )

        optimizer.zero_grad()

        outputs = model(
            inputs,
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        train_correct += (predictions == labels).sum().item()

        train_total += labels.size(0)

    train_loss = running_loss / len(train_loader)

    train_accuracy = train_correct / train_total

    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0

    val_labels = []

    val_predictions = []

    with torch.no_grad():

        for (
            mfcc,
            mel,
            chroma,
            labels,
        ) in val_loader:

            mfcc = mfcc.to(device)

            mel = mel.to(device)

            chroma = chroma.to(device)

            labels = labels.to(device)

            inputs = convert_features(
                mfcc,
                mel,
                chroma,
            )

            outputs = model(
                inputs,
            )

            loss = criterion(
                outputs,
                labels,
            )

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            val_correct += (predictions == labels).sum().item()

            val_total += labels.size(0)

            val_labels.extend(labels.cpu().numpy())

            val_predictions.extend(predictions.cpu().numpy())

    val_loss /= len(val_loader)

    val_accuracy = val_correct / val_total

    val_labels = np.array(val_labels)

    val_predictions = np.array(val_predictions)

    macro_f1, weighted_f1, balanced_acc = compute_metrics(
        val_labels,
        val_predictions,
    )

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    scheduler.step(macro_f1)

    # ======================================================
    # Save best model
    # ======================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print(f"\nBest model saved: " f"Macro F1 = {macro_f1:.4f}")

    else:

        epochs_without_improvement += 1

    # ======================================================
    # History
    # ======================================================

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
        }
    )

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy*100:.2f}% | "
        f"Val Acc: {val_accuracy*100:.2f}% | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Balanced Acc: {balanced_acc:.4f}"
    )

    # ======================================================
    # Early stopping
    # ======================================================

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ==========================================================
# Save history
# ==========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False,
)


# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)

print("EFFICIENTNET-B0 TRAINING FINISHED")

print("=" * 60)

print(f"Best Validation Macro F1: " f"{best_macro_f1:.4f}")

print("\nModel saved to:")

print(MODEL_PATH)

print("\nHistory saved to:")

print(HISTORY_PATH)
