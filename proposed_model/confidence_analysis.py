import torch
import torch.nn.functional as F
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import EXPERIMENT_NAME

SAVE_DIR = os.path.join("results", EXPERIMENT_NAME)
MODEL_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################

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

############################################

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

############################################

results = []

correct_conf = []
wrong_conf = []

class_conf = {}

############################################

with torch.no_grad():

    for mfcc, mel, chroma, label in loader:

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)
        label = label.to(device)

        output, _, _ = model(mfcc, mel, chroma)

        prob = F.softmax(output, dim=1)

        conf, pred = torch.max(prob, 1)

        conf = conf.item()

        pred = pred.item()

        true = label.item()

        results.append(
            [
                encoder.classes_[true],
                encoder.classes_[pred],
                conf,
                pred == true,
            ]
        )

        if pred == true:
            correct_conf.append(conf)
        else:
            wrong_conf.append(conf)

        class_name = encoder.classes_[pred]

        class_conf.setdefault(class_name, []).append(conf)

############################################

results_df = pd.DataFrame(
    results,
    columns=[
        "True",
        "Prediction",
        "Confidence",
        "Correct",
    ],
)

results_df.to_csv(
    "results/confidence_analysis.csv",
    index=False,
)

print()

print("Average confidence (Correct):")

print(sum(correct_conf) / len(correct_conf))

print()

print("Average confidence (Wrong):")

print(sum(wrong_conf) / len(wrong_conf))

print()

print("Average confidence by predicted class")

for k, v in class_conf.items():

    print(f"{k:20s}{sum(v)/len(v):.3f}")
