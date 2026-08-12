import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)

from sklearn.preprocessing import label_binarize

import os

from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("features/labels.csv")

encoder = LabelEncoder()

df["label_encoded"] = encoder.fit_transform(df["label"])

_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"],
)

dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root="features",
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)


model = PediaLungXAI(num_classes=7).to(device)

torch.load(
    os.path.join(
        MODEL_DIR,
        f"{EXPERIMENT_NAME}_best.pth",
    ),
    map_location=device,
)


model.eval()


all_probs = []
all_labels = []

with torch.no_grad():

    for mfcc, mel, chroma, label in loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        output, _, _ = model(mfcc, mel, chroma)

        prob = F.softmax(output, dim=1)

        all_probs.append(prob.cpu().numpy())
        all_labels.append(label.numpy())


all_probs = np.concatenate(all_probs)

all_labels = np.concatenate(all_labels)

binary_labels = label_binarize(
    all_labels,
    classes=np.arange(7),
)

plt.figure(figsize=(8, 6))

for i in range(7):

    fpr, tpr, _ = roc_curve(binary_labels[:, i], all_probs[:, i])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{encoder.classes_[i]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "roc_curve.png"),
    dpi=300,
    bb0x_inches="tight",
)

plt.close()

plt.figure(figsize=(8, 6))

for i in range(7):

    precision, recall, _ = precision_recall_curve(binary_labels[:, i], all_probs[:, i])

    plt.plot(recall, precision, label=encoder.classes_[i])

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves")

plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "precision_recall_curve.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("ROC and PR curves saved.")
