import os
import sys

# ==========================================================
# Reduce CPU / memory pressure
# ==========================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score

from torch.utils.data import DataLoader, WeightedRandomSampler

from multifeature_dataset import MultiFeatureDataset
from focal_loss import FocalLoss

# ==========================================================
# Configuration
# ==========================================================

FEATURE_ROOT = os.path.join(PROJECT_ROOT, "features")
LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_models",
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "resnet18_best.pth")

HISTORY_PATH = os.path.join(RESULT_DIR, "history.csv")

CLASS_NAMES_PATH = os.path.join(RESULT_DIR, "class_names.json")


BATCH_SIZE = 8
NUM_EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

FOCAL_GAMMA = 2.0

EARLY_STOPPING_PATIENCE = 10

TARGET_HEIGHT = 128
TARGET_WIDTH = 259


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("========================================")
print("ResNet-18 COMPARISON MODEL")
print("========================================")
print("Device:", device)
print("Feature root:", FEATURE_ROOT)


# ==========================================================
# ResNet-18 Implementation
# ==========================================================


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:

            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)

        self.layer2 = self._make_layer(128, 2, stride=2)

        self.layer3 = self._make_layer(256, 2, stride=2)

        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(BasicBlock(self.in_channels, out_channels, stride))

        self.in_channels = out_channels

        for _ in range(1, blocks):

            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# ==========================================================
# Feature Conversion
# ==========================================================


def convert_features(mfcc, mel, chroma):

    # ------------------------------------------------------
    # Each input:
    #
    # MFCC    : [B, 1, 40, 94]
    # Mel     : [B, 1, 128, 259]
    # Chroma  : [B, 1, 12, 259]
    #
    # Convert all to:
    #
    # [B, 1, 128, 259]
    # ------------------------------------------------------

    mfcc = torch.nn.functional.interpolate(
        mfcc, size=(TARGET_HEIGHT, TARGET_WIDTH), mode="bilinear", align_corners=False
    )

    mel = torch.nn.functional.interpolate(
        mel, size=(TARGET_HEIGHT, TARGET_WIDTH), mode="bilinear", align_corners=False
    )

    chroma = torch.nn.functional.interpolate(
        chroma, size=(TARGET_HEIGHT, TARGET_WIDTH), mode="bilinear", align_corners=False
    )

    # ------------------------------------------------------
    # Stack three feature types as channels
    # ------------------------------------------------------

    x = torch.cat([mfcc, mel, chroma], dim=1)

    return x


# ==========================================================
# Metrics
# ==========================================================


def compute_metrics(y_true, y_pred):

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return (macro_f1, weighted_f1, balanced_acc)


# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

num_classes = len(label_encoder.classes_)

print("\nClasses:")

for i, cls in enumerate(label_encoder.classes_):

    print(i, ":", cls)


# Save class names

with open(CLASS_NAMES_PATH, "w") as f:

    json.dump(label_encoder.classes_.tolist(), f, indent=4)


# ==========================================================
# EXACT SAME DATA SPLIT
# ==========================================================

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


# ==========================================================
# Datasets
# ==========================================================

train_dataset = MultiFeatureDataset(
    dataframe=train_df, feature_root=FEATURE_ROOT, train=True
)


val_dataset = MultiFeatureDataset(
    dataframe=val_df, feature_root=FEATURE_ROOT, train=False
)


test_dataset = MultiFeatureDataset(
    dataframe=test_df, feature_root=FEATURE_ROOT, train=False
)


# ==========================================================
# Weighted Random Sampler
# ==========================================================

class_counts = train_df["label_encoded"].value_counts().sort_index()

class_weights = 1.0 / class_counts

sample_weights = train_df["label_encoded"].map(class_weights)

sample_weights = torch.DoubleTensor(sample_weights.values)

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)


print("\nWeightedRandomSampler: ENABLED")


# ==========================================================
# Data Loaders
# ==========================================================

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
)


val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ==========================================================
# Model
# ==========================================================

model = ResNet18(num_classes=num_classes).to(device)


print("\n==============================")
print("MODEL")
print("==============================")

print(model)


total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nTotal Parameters:", f"{total_params:,}")

print("Trainable Parameters:", f"{trainable_params:,}")


# ==========================================================
# Loss / Optimizer
# ==========================================================

criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=None)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
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

    for mfcc, mel, chroma, labels in train_loader:

        mfcc = mfcc.to(device)

        mel = mel.to(device)

        chroma = chroma.to(device)

        labels = labels.to(device)

        # Convert to 3-channel input

        inputs = convert_features(mfcc, mel, chroma)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

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

        for mfcc, mel, chroma, labels in val_loader:

            mfcc = mfcc.to(device)

            mel = mel.to(device)

            chroma = chroma.to(device)

            labels = labels.to(device)

            inputs = convert_features(mfcc, mel, chroma)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (predictions == labels).sum().item()

            val_total += labels.size(0)

            val_labels.extend(labels.cpu().numpy())

            val_predictions.extend(predictions.cpu().numpy())

    val_loss /= len(val_loader)

    val_accuracy = val_correct / val_total

    val_labels = np.array(val_labels)

    val_predictions = np.array(val_predictions)

    macro_f1, weighted_f1, balanced_acc = compute_metrics(val_labels, val_predictions)

    scheduler.step(macro_f1)

    # ======================================================
    # Save best model
    # ======================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        epochs_without_improvement = 0

        torch.save(model.state_dict(), MODEL_PATH)

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

history_df.to_csv(HISTORY_PATH, index=False)


# ==========================================================
# Finished
# ==========================================================

print("\n========================================")
print("RESNET-18 TRAINING FINISHED")
print("========================================")

print(f"Best Validation Macro F1: " f"{best_macro_f1:.4f}")

print("Model saved to:")

print(MODEL_PATH)

print("History saved to:")

print(HISTORY_PATH)
