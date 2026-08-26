import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
# Load model
#####################################################

model = PediaLungXAI(num_classes=7).to(device)

torch.load(
    os.path.join(
        MODEL_DIR,
        f"{EXPERIMENT_NAME}_best.pth",
    ),
    map_location=device,
)

model.eval()

#####################################################
# Dataset
#####################################################

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

#####################################################
# Collect fusion weights
#####################################################

weights = []

with torch.no_grad():

    for mfcc, mel, chroma, _ in loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        _, fusion, _ = model(
            mfcc,
            mel,
            chroma,
        )

        weights.append(fusion.cpu().numpy())

weights = np.concatenate(weights)

avg = weights.mean(axis=0)

print("\nAverage Fusion Weights")

print(f"MFCC   : {avg[0]:.4f}")

print(f"Mel    : {avg[1]:.4f}")

print(f"Chroma : {avg[2]:.4f}")

#####################################################
# Plot
#####################################################

plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["MFCC", "Mel", "Chroma"],
    avg,
)

for b, v in zip(bars, avg):

    plt.text(
        b.get_x() + b.get_width() / 2,
        v + 0.005,
        f"{v:.3f}",
        ha="center",
        fontsize=11,
    )

plt.ylim(0, 0.5)

plt.ylabel("Average Weight")

plt.title("Average Adaptive Fusion Weights")

os.makedirs("results", exist_ok=True)

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "fusion_weights.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved to results/fusion_weights.png")
