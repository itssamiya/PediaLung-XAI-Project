import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from multibranch_model import PediaLungXAI
from multifeature_dataset import MultiFeatureDataset
from gradcam import GradCAM, visualize_gradcam

from config import MODEL_CONFIG
from config import EXPERIMENT_NAME
from config import MODEL_DIR
from config import SAVE_DIR

# ==========================================================
# Configuration
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Testing Grad-CAM")
print("=" * 60)

print("Device:", device)
print("Experiment:", EXPERIMENT_NAME)


# ==========================================================
# Load model
# ==========================================================

print("\nLoading model...")

model = PediaLungXAI(
    num_classes=7,
    **MODEL_CONFIG["proposed"],
).to(device)

checkpoint_path = os.path.join(
    MODEL_DIR,
    f"{EXPERIMENT_NAME}_best.pth",
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
    )
)

model.eval()

print("Model loaded successfully.")


# ==========================================================
# Label Encoder
# ==========================================================

df = pd.read_csv("features/labels.csv")

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

print("\nClasses:")

for i, name in enumerate(label_encoder.classes_):
    print(i, "->", name)


# ==========================================================
# Create same test split
# ==========================================================

_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_encoded"],
)

print("\nTest samples:", len(test_df))


# ==========================================================
# Dataset
# ==========================================================

dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root="features",
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)


# ==========================================================
# Select one sample
# ==========================================================

mfcc, mel, chroma, label = next(iter(loader))

mfcc = mfcc.to(device)
mel = mel.to(device)
chroma = chroma.to(device)
label = label.to(device)


true_class = label.item()

print("\nTrue Label:")
print(label_encoder.classes_[true_class])


# ==========================================================
# Model Prediction
# ==========================================================

with torch.no_grad():

    output, fusion_weights, attention_map, feature_map = model(
        mfcc,
        mel,
        chroma,
    )

probabilities = torch.softmax(output, dim=1)

predicted_class = torch.argmax(
    probabilities,
    dim=1,
).item()

confidence = probabilities[0, predicted_class].item()

print("\nPrediction:")
print(label_encoder.classes_[predicted_class])

print(f"Confidence: {confidence * 100:.2f}%")


# ==========================================================
# Top-3 Predictions
# ==========================================================

top3 = torch.topk(
    probabilities,
    k=3,
    dim=1,
)

top_predictions = []

for prob, cls in zip(
    top3.values[0],
    top3.indices[0],
):

    top_predictions.append(
        (
            label_encoder.classes_[cls.item()],
            prob.item(),
        )
    )


print("\nTop-3 Predictions:")

for name, prob in top_predictions:
    print(f"{name:20s} {prob * 100:.2f}%")


# ==========================================================
# Initialize Grad-CAM
# ==========================================================

print("\nInitializing Grad-CAM...")

gradcam = GradCAM(model)

print("Grad-CAM initialized.")


# ==========================================================
# Generate Heatmap
# ==========================================================

print("\nGenerating heatmap...")

heatmap = gradcam.generate(
    mfcc,
    mel,
    chroma,
    predicted_class,
)

heatmap = cv2.GaussianBlur(
    heatmap,
    (11, 11),
    0,
)

print("Heatmap generated.")


# ==========================================================
# Save Grad-CAM Figure
# ==========================================================

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)

save_path = os.path.join(
    SAVE_DIR,
    "test_gradcam.png",
)


visualize_gradcam(
    mel,
    heatmap,
    save_path,
    label_encoder.classes_[true_class],
    label_encoder.classes_[predicted_class],
    confidence,
    top_predictions,
)


print("\n" + "=" * 60)
print("Grad-CAM test completed.")
print("Saved image:")
print(save_path)
print("=" * 60)
