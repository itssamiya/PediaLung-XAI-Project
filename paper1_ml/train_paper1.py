import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset_paper1 import RespiratoryDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset_paper1 import RespiratoryDataset
from model_paper1 import LightweightCNN

# =====================================================
# Paths
# =====================================================

FEATURE_DIR = "features/mfcc"
LABEL_CSV = "features/labels.csv"

# =====================================================
# Load Labels
# =====================================================

df = pd.read_csv(LABEL_CSV)

print(df.head())

print("\nDataset Information")
print(df.info())

print("\nClass Distribution")
print(df["label"].value_counts())

# =====================================================
# Encode Labels
# =====================================================

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

print("\nEncoded Labels")
print(df.head())

# =====================================================
# Train-Test Split
# =====================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"]
)

##################################################
# Compute Class Weights
##################################################

class_counts = train_df["label_encoded"].value_counts().sort_index()

class_weights = 1.0 / class_counts.values

class_weights = class_weights / class_weights.sum()

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass Weights")

print(class_weights)

print("\nTraining Samples:", len(train_df))
print("Testing Samples:", len(test_df))

print("\nTraining Class Distribution")
print(train_df["label"].value_counts())

print("\nTesting Class Distribution")
print(test_df["label"].value_counts())

# =====================================================
# Create Dataset
# =====================================================

train_dataset = RespiratoryDataset(
    dataframe=train_df,
    feature_dir=FEATURE_DIR
)

test_dataset = RespiratoryDataset(
    dataframe=test_df,
    feature_dir=FEATURE_DIR
)

# =====================================================
# Create DataLoader
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# =====================================================
# Save Label Mapping
# =====================================================

label_mapping = dict(
    zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )
)

print("\nLabel Mapping")

for disease, idx in label_mapping.items():
    print(f"{disease} --> {idx}")

    # =====================================================
# Verify DataLoader
# =====================================================

for mfccs, labels in train_loader:

    print("MFCC Batch Shape :", mfccs.shape)
    print("Label Shape      :", labels.shape)
    print("Labels           :", labels)

    break

##############################################
# Device
##############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)
print("\nUsing Device:", device)


##############################################
# Model
##############################################

model = LightweightCNN(num_classes=7).to(device)

print(model)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
NUM_EPOCHS = 20

#########################################################
# Create folders
#########################################################

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

#########################################################
# Best Model
#########################################################

best_accuracy = 0.0

#########################################################
# Training Loop
#########################################################

for epoch in range(NUM_EPOCHS):

    ###################################
    # Training
    ###################################

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    ###################################
    # Validation
    ###################################

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    ###################################
    # Save Best Model
    ###################################

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "saved_models/baseline_cnn.pth"
        )

    
    epoch_loss = running_loss / len(train_loader)
    print(
    f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
    f"Loss: {epoch_loss:.4f} | "
    f"Train Acc: {train_accuracy:.2f}% | "
    f"Val Acc: {val_accuracy:.2f}%"
)

print("\nTraining Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")