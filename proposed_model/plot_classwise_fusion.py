import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from config import EXPERIMENT_NAME

SAVE_DIR = os.path.join("results", EXPERIMENT_NAME)
MODEL_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
# Load Model
#####################################################

model = PediaLungXAI(num_classes=7).to(device)

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            f"{EXPERIMENT_NAME}_best.pth",
        ),
        map_location=device,
    )
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
    batch_size=1,
    shuffle=False,
)

#####################################################
# Collect Weights
#####################################################

class_weights = {}

with torch.no_grad():

    for mfcc, mel, chroma, label in loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        _, fusion, _ = model(
            mfcc,
            mel,
            chroma,
        )

        label_name = encoder.classes_[label.item()]

        if label_name not in class_weights:
            class_weights[label_name] = []

        class_weights[label_name].append(fusion.squeeze().cpu().numpy())

#####################################################
# Average
#####################################################

labels = []

mfcc_avg = []

mel_avg = []

chroma_avg = []

print()

for cls in encoder.classes_:

    arr = np.array(class_weights[cls])

    avg = arr.mean(axis=0)

    labels.append(cls)

    mfcc_avg.append(avg[0])

    mel_avg.append(avg[1])

    chroma_avg.append(avg[2])

    print(cls, avg)

#####################################################
# Plot
#####################################################

x = np.arange(len(labels))

width = 0.25

plt.figure(figsize=(10, 5))

plt.bar(
    x - width,
    mfcc_avg,
    width,
    label="MFCC",
)

plt.bar(
    x,
    mel_avg,
    width,
    label="Mel",
)

plt.bar(
    x + width,
    chroma_avg,
    width,
    label="Chroma",
)

plt.xticks(
    x,
    labels,
    rotation=20,
)

plt.ylabel("Average Fusion Weight")

plt.title("Class-wise Adaptive Feature Fusion")

plt.legend()

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    os.path.join(SAVE_DIR, "classwise_fushion_weights.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved.")
