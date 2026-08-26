import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config as cfg


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, WeightedRandomSampler

from multifeature_dataset import MultiFeatureDataset
from multibranch_model import PediaLungXAI
from focal_loss import FocalLoss


from utils.early_stopping import EarlyStopping

early_stopping = EarlyStopping(patience=cfg.EARLY_STOPPING_PATIENCE)
from config import MODEL_CONFIG
from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

# ==========================================================
# Paths
# ==========================================================

FEATURE_ROOT = "features"
LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

print("LABEL_CSV =", LABEL_CSV)
print("Exists:", os.path.exists(LABEL_CSV))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ==========================================================
# Load Labels
# ==========================================================

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

print(df.head())


#########################################################
# Compute Macro F1 without sklearn
#########################################################


def compute_macro_f1(y_true, y_pred, num_classes):

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


#########################################################
# Compute Weighted F1
#########################################################


def compute_weighted_f1(y_true, y_pred, num_classes):

    total = len(y_true)

    weighted = 0

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
# Train / Validation / Test Split
# ==========================================================

trainval_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_encoded"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.125,  # 10% of total data
    random_state=42,
    stratify=trainval_df["label_encoded"],
)

print(train_df["label_encoded"].value_counts().sort_index())
print(label_encoder.classes_)

print("Training Samples   :", len(train_df))
print("Validation Samples :", len(val_df))
print("Testing Samples    :", len(test_df))

# ==========================================================
# Class Weights
# ==========================================================

class_counts = train_df["label_encoded"].value_counts().sort_index()

# samples_per_class = class_counts.tolist()

# class_weights = len(train_df) / (len(class_counts) * class_counts.values)

# class_weights = torch.tensor(
# class_weights,
# dtype=torch.float32,
# device=device,
# )

print("Class Counts:")
print(class_counts)

# print("\nClass Weights:")
# print(class_weights)
# ==========================================================
# Dataset
# ==========================================================

train_dataset = MultiFeatureDataset(
    dataframe=train_df,
    feature_root="features",
    train=True,
)

# ==========================================================
# Weighted Random Sampler
# ==========================================================

if cfg.USE_WEIGHTED_SAMPLER:

    class_counts = train_df["label_encoded"].value_counts().sort_index()

    class_weights = 1.0 / class_counts

    sample_weights = train_df["label_encoded"].map(class_weights)

    sample_weights = torch.DoubleTensor(sample_weights.values)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("\nWeightedRandomSampler: ENABLED")

else:

    sampler = None

    print("\nWeightedRandomSampler: DISABLED")

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
# Data Loaders
# ==========================================================


train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
# ==========================================================
# Model
# ==========================================================

model = PediaLungXAI(
    num_classes=len(label_encoder.classes_), **MODEL_CONFIG[cfg.MODEL_NAME]
)

print(model)


criterion = FocalLoss(gamma=cfg.FOCAL_GAMMA, alpha=None)

optimizer = optim.Adam(
    model.parameters(),
    lr=cfg.LEARNING_RATE,
    weight_decay=cfg.WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
)

epochs = cfg.NUM_EPOCHS

# ==========================================================
# Create folders
# ==========================================================

best_macro_f1 = 0
best_accuracy = 0

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "macro_f1": [],
    "weighted_f1": [],
}

# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(epochs):

    # -----------------------------
    # Training
    # -----------------------------

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

        outputs, weights, attention, _ = model(mfcc, mel, chroma)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100.0 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------

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

            outputs, weights, attention, _ = model(mfcc, mel, chroma)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)

        val_accuracy = 100.0 * val_correct / val_total

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        macro_f1 = compute_macro_f1(all_labels, all_preds, len(label_encoder.classes_))

        weighted_f1 = compute_weighted_f1(
            all_labels, all_preds, len(label_encoder.classes_)
        )

    scheduler.step(macro_f1)

    #########################################################
    # Save Best Model (Macro F1)
    #########################################################

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_best.pth"),
        )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_best_acc.pth"),
        )

    #########################################################
    # Early Stopping (Macro F1)
    #########################################################

    if early_stopping.step(macro_f1):
        print("\nEarly stopping triggered.")
        break

    history["epoch"].append(epoch + 1)

    history["train_loss"].append(train_loss)

    history["val_loss"].append(val_loss)

    history["train_acc"].append(train_accuracy)

    history["val_acc"].append(val_accuracy)

    history["macro_f1"].append(macro_f1)

    history["weighted_f1"].append(weighted_f1)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted F1: {weighted_f1:.4f}"
    )

# ==========================================================
# Save Training History
# ==========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(SAVE_DIR, "history.csv"),
    index=False,
)


print("\n========================================")
print("Training Finished!")
print("========================================")

print(f"Best Validation Macro F1: {best_macro_f1:.4f}")

print("Model saved to:")
print(os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_best.pth"))

print("Training history saved to:")
print(os.path.join(SAVE_DIR, "history.csv"))
