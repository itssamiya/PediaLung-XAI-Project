import pandas as pd
import matplotlib.pyplot as plt

import os

from config import EXPERIMENT_NAME

SAVE_DIR = os.path.join("results", EXPERIMENT_NAME)
MODEL_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load confidence analysis
df = pd.read_csv(os.path.join(SAVE_DIR, "confidence_analysis.csv"))

##################################################
# Histogram of confidence
##################################################

plt.figure(figsize=(8, 5))

plt.hist(df["Confidence"], bins=20)

plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Distribution of Prediction Confidence")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "confidence_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
)

plt.close()

##################################################
# Average confidence by predicted class
##################################################

avg = df.groupby("Prediction")["Confidence"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))

plt.bar(avg.index, avg.values)

plt.ylabel("Average Confidence")

plt.title("Average Prediction Confidence by Class")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "confidence_by_class.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Confidence figures saved.")
