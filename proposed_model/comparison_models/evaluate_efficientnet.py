import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# Disable MKLDNN to avoid CPU primitive errors with EfficientNet
torch.backends.mkldnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F

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
# EXACT SAME AS TRAINING
# ============================================================


def convert_features(
    mfcc,
    mel,
    chroma,
):

    # --------------------------------------------------------
    # MFCC
    # [B, 1, 40, 94]
    # ->
    # [B, 1, 128, 259]
    # --------------------------------------------------------

    mfcc = F.interpolate(
        mfcc,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # --------------------------------------------------------
    # Mel
    # [B, 1, 128, 259]
    # --------------------------------------------------------

    mel = F.interpolate(
        mel,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # --------------------------------------------------------
    # Chroma
    # [B, 1, 12, 259]
    # ->
    # [B, 1, 128, 259]
    # --------------------------------------------------------

    chroma = F.interpolate(
        chroma,
        size=(
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # --------------------------------------------------------
    # Combine as three channels
    #
    # Channel 0 = MFCC
    # Channel 1 = Mel
    # Channel 2 = Chroma
    #
    # [B, 3, 128, 259]
    # --------------------------------------------------------

    x = torch.cat(
        [
            mfcc,
            mel,
            chroma,
        ],
        dim=1,
    )

    return x


# ============================================================
# EFFICIENTNET-B0 MODEL
# EXACT SAME ARCHITECTURE AS TRAINING
# ============================================================


class EfficientNetB0Model(nn.Module):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()

        self.backbone = efficientnet_b0(
            weights=None,
        )

        feature_dim = self.backbone.classifier[-1].in_features

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


# ============================================================
# LOAD LABEL DATA
# ============================================================

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


# ============================================================
# EXACT SAME DATA SPLIT AS TRAINING
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


# ============================================================
# TEST DATASET
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


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

model = EfficientNetB0Model(
    num_classes=num_classes,
).to(DEVICE)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(state_dict)

model.eval()

print("EfficientNet-B0 loaded successfully.")


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


print(
    "\nTotal Parameters     :",
    f"{total_params:,}",
)

print(
    "Trainable Parameters :",
    f"{trainable_params:,}",
)


# ============================================================
# TEST INFERENCE
# ============================================================

print("\n========================================")
print("RUNNING EFFICIENTNET-B0 TEST INFERENCE")
print("========================================")


y_true = []
y_pred = []
y_prob = []


processed = 0


with torch.no_grad():

    for (
        mfcc,
        mel,
        chroma,
        labels,
    ) in test_loader:

        mfcc = mfcc.to(DEVICE)

        mel = mel.to(DEVICE)

        chroma = chroma.to(DEVICE)

        labels = labels.to(DEVICE)

        # EXACT SAME FEATURE CONVERSION
        # USED DURING TRAINING

        inputs = convert_features(
            mfcc,
            mel,
            chroma,
        )

        outputs = model(inputs)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(predictions.cpu().numpy())

        y_prob.extend(probabilities.cpu().numpy())

        processed += labels.size(0)

        if processed % 100 < labels.size(0):

            print(f"Processed " f"{processed}/" f"{len(test_dataset)}")


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)


balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)


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

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)


# ============================================================
# PRINT RESULTS
# ============================================================

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


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "metrics.txt",
)


with open(
    metrics_path,
    "w",
) as f:

    f.write("EFFICIENTNET-B0 TEST EVALUATION\n")

    f.write("=" * 60 + "\n\n")

    f.write(f"Test Samples       : " f"{len(test_dataset)}\n")

    f.write(f"Total Parameters   : " f"{total_params:,}\n")

    f.write(f"Trainable Parameters : " f"{trainable_params:,}\n")

    f.write(f"Accuracy           : " f"{accuracy:.4f}\n")

    f.write(f"Precision          : " f"{precision:.4f}\n")

    f.write(f"Recall             : " f"{recall:.4f}\n")

    f.write(f"Weighted F1        : " f"{weighted_f1:.4f}\n")

    f.write(f"Macro F1           : " f"{macro_f1:.4f}\n")

    f.write(f"Balanced Accuracy  : " f"{balanced_accuracy:.4f}\n")


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt",
)


with open(
    report_path,
    "w",
) as f:

    f.write(report)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.csv",
)


cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
)


cm_df.to_csv(cm_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test_df.copy()


predictions["true_class"] = label_encoder.inverse_transform(y_true)


predictions["predicted_class"] = label_encoder.inverse_transform(y_pred)


predictions["confidence"] = np.max(y_prob, axis=1)


predictions_path = os.path.join(
    RESULT_DIR,
    "predictions.csv",
)


predictions.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print("\nResults saved to:")

print(metrics_path)

print(report_path)

print(cm_path)

print(predictions_path)


print("\n========================================")
print("EFFICIENTNET-B0 EVALUATION FINISHED")
print("========================================")
