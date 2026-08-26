import os
import sys

# Add proposed_model directory to Python path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from torch.utils.data import DataLoader, WeightedRandomSampler

import config as cfg

from multifeature_dataset import MultiFeatureDataset
from multibranch_model import PediaLungXAI
from focal_loss import FocalLoss


# ==========================================================
# Settings
# ==========================================================

EXPERIMENT_NAME = "hierarchical_binary"

FEATURE_ROOT = "features"
LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

RESULT_DIR = os.path.join("results", EXPERIMENT_NAME)
MODEL_DIR = "saved_models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
PATIENCE = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(LABEL_CSV)

print("\nOriginal Distribution:")
print(df["label"].value_counts())


# ==========================================================
# Create Binary Labels
# ==========================================================

df["binary_label"] = df["label"].apply(
    lambda x: "Normal" if x == "Normal" else "Abnormal"
)

df["label_encoded"] = df["binary_label"].map(
    {
        "Normal": 0,
        "Abnormal": 1,
    }
)

print("\nBinary Distribution:")
print(df["binary_label"].value_counts())


# ==========================================================
# Train / Validation / Test Split
# ==========================================================

trainval_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df["binary_label"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.111111,
    random_state=42,
    stratify=trainval_df["binary_label"],
)

print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("\nTraining:")
print(train_df["binary_label"].value_counts())

print("\nValidation:")
print(val_df["binary_label"].value_counts())

print("\nTesting:")
print(test_df["binary_label"].value_counts())


# ==========================================================
# Dataset
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

print("\nClass Counts:")
print(class_counts)

class_weights = 1.0 / class_counts

sample_weights = train_df["label_encoded"].map(
    class_weights
)

sample_weights = torch.DoubleTensor(
    sample_weights.values
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

print("\nWeightedRandomSampler: ENABLED")


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
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

model = PediaLungXAI(
    num_classes=2,
    **cfg.MODEL_CONFIG["proposed"],
).to(device)

print("\n==============================")
print("BINARY MODEL")
print("==============================")
print("Architecture : proposed")
print("Classes      : 2")
print("Loss         : Focal Loss")
print("Gamma        :", FOCAL_GAMMA)


# ==========================================================
# Loss / Optimizer
# ==========================================================

criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    alpha=None,
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

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

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    model.train()

    running_loss = 0.0
    train_true = []
    train_pred = []

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

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        train_true.extend(
            labels.cpu().numpy()
        )

        train_pred.extend(
            predictions.cpu().numpy()
        )

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = accuracy_score(
        train_true,
        train_pred,
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_true = []
    val_pred = []

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

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            val_true.extend(
                labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )

    val_loss /= len(val_loader)

    val_accuracy = accuracy_score(
        val_true,
        val_pred,
    )

    val_macro_f1 = f1_score(
        val_true,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_balanced_accuracy = (
        balanced_accuracy_score(
            val_true,
            val_pred,
        )
    )

    scheduler.step(val_macro_f1)

    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1

        epochs_without_improvement = 0

        model_path = os.path.join(
            MODEL_DIR,
            f"{EXPERIMENT_NAME}_best.pth",
        )

        torch.save(
            model.state_dict(),
            model_path,
        )

        print(
            f"\nBest model saved: "
            f"Macro F1 = {best_macro_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

    # ------------------------------------------------------
    # History
    # ------------------------------------------------------

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy":
                val_balanced_accuracy,
        }
    )

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy*100:.2f}% | "
        f"Val Acc: {val_accuracy*100:.2f}% | "
        f"Macro F1: {val_macro_f1:.4f} | "
        f"Balanced Acc: {val_balanced_accuracy:.4f}"
    )

    # ------------------------------------------------------
    # Early Stopping
    # ------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==========================================================
# Save History
# ==========================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    RESULT_DIR,
    "history.csv",
)

history_df.to_csv(
    history_path,
    index=False,
)


# ==========================================================
# Finished
# ==========================================================

print("\n========================================")
print("Binary Training Finished")
print("========================================")

print(
    f"Best Validation Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    "Model saved to:",
    os.path.join(
        MODEL_DIR,
        f"{EXPERIMENT_NAME}_best.pth",
    ),
)

print(
    "History saved to:",
    history_path,
)