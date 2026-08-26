import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os

from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################
# Load Model
##########################################################

model = PediaLungXAI(num_classes=7).to(device)

torch.load(
    os.path.join(
        MODEL_DIR,
        f"{EXPERIMENT_NAME}_best.pth",
    ),
    map_location=device,
)


model.eval()

##########################################################
# Dataset
##########################################################

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

##########################################################
# Prediction
##########################################################

y_true = []

y_pred = []

with torch.no_grad():

    for mfcc, mel, chroma, label in loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        outputs, _, _ = model(
            mfcc,
            mel,
            chroma,
        )

        prediction = outputs.argmax(dim=1)

        y_true.extend(label.numpy())

        y_pred.extend(prediction.cpu().numpy())

##########################################################
# Confusion Matrix
##########################################################

cm = confusion_matrix(
    y_true,
    y_pred,
)

plt.figure(figsize=(8, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix")

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    os.path.join(SAVE_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Confusion matrix saved.")
