import os
import numpy as np
import pandas as pd
import torch

import config as cfg

from multibranch_model import PediaLungXAI

from utils.preprocessing import (
    load_audio,
    wavelet_denoise,
    normalize_length,
    extract_mfcc,
    extract_mel,
    extract_chroma,
)

from config import MODEL_DIR, EXPERIMENT_NAME

# ============================================================
# SETTINGS
# ============================================================

TEST_DIR = r"D:\PediaLung-XAI\data\test2022_wav"

NUM_SAMPLES = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FUSION WEIGHT ANALYSIS")
print("=" * 70)

print("Device:", device)
print("Experiment:", EXPERIMENT_NAME)
print("Model:", cfg.MODEL_NAME)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = PediaLungXAI(
    num_classes=7,
    **cfg.MODEL_CONFIG[cfg.MODEL_NAME],
).to(device)

checkpoint_path = os.path.join(
    MODEL_DIR,
    f"{EXPERIMENT_NAME}_best.pth",
)

print("Checkpoint:", checkpoint_path)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
    )
)

model.eval()

print("Model loaded.")


# ============================================================
# FIND WAV FILES
# ============================================================

wav_files = []

for root, dirs, files in os.walk(TEST_DIR):

    for file in files:

        if file.lower().endswith(".wav"):

            wav_files.append(os.path.join(root, file))


wav_files = sorted(wav_files)

print("\nTotal WAV files found:", len(wav_files))

if len(wav_files) == 0:

    raise RuntimeError(f"No WAV files found in:\n{TEST_DIR}")


wav_files = wav_files[:NUM_SAMPLES]

print(f"Analyzing first {len(wav_files)} samples...")


# ============================================================
# FEATURE PREPROCESSING
# ============================================================


def preprocess(wav_path):

    signal, sr = load_audio(wav_path)

    signal = wavelet_denoise(signal)

    signal = normalize_length(
        signal,
        sr,
    )

    mfcc = extract_mfcc(
        signal,
        sr,
    )

    mel = extract_mel(
        signal,
        sr,
    )

    chroma = extract_chroma(
        signal,
        sr,
    )

    return mfcc, mel, chroma


# ============================================================
# STORAGE
# ============================================================

results = []


# ============================================================
# ANALYSIS
# ============================================================

for sample_number, wav_path in enumerate(
    wav_files,
    start=1,
):

    try:

        mfcc, mel, chroma = preprocess(wav_path)

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        mfcc = (
            torch.tensor(
                mfcc,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mel = (
            torch.tensor(
                mel,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        chroma = (
            torch.tensor(
                chroma,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mfcc = mfcc.to(device)
        mel = mel.to(device)
        chroma = chroma.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.no_grad():

            outputs, fusion_weights, _, _ = model(
                mfcc,
                mel,
                chroma,
            )

        # ----------------------------------------------------
        # Fusion weights
        # ----------------------------------------------------

        weights = fusion_weights[0].detach().cpu().numpy()

        mfcc_weight = float(weights[0])
        mel_weight = float(weights[1])
        chroma_weight = float(weights[2])

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted_class = int(probabilities.argmax(dim=1).item())

        confidence = float(probabilities[0, predicted_class].item())

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append(
            {
                "sample": sample_number,
                "file": os.path.basename(wav_path),
                "mfcc_weight": mfcc_weight,
                "mel_weight": mel_weight,
                "chroma_weight": chroma_weight,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
        )

        print(
            f"[{sample_number:03d}/{len(wav_files):03d}] "
            f"MFCC={mfcc_weight:.4f} | "
            f"Mel={mel_weight:.4f} | "
            f"Chroma={chroma_weight:.4f} | "
            f"Class={predicted_class} | "
            f"Conf={confidence:.2%}"
        )

    except Exception as e:

        print(f"[{sample_number:03d}] ERROR: " f"{os.path.basename(wav_path)}")

        print("   ", e)


# ============================================================
# DATAFRAME
# ============================================================

if len(results) == 0:

    raise RuntimeError("No samples were successfully analyzed.")


df = pd.DataFrame(results)


# ============================================================
# STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("FUSION WEIGHT STATISTICS")
print("=" * 70)


for feature in [
    "mfcc_weight",
    "mel_weight",
    "chroma_weight",
]:

    values = df[feature]

    print(f"\n{feature.upper()}")

    print(f"Mean   : {values.mean():.4f}")

    print(f"Median : {values.median():.4f}")

    print(f"Std    : {values.std():.4f}")

    print(f"Min    : {values.min():.4f}")

    print(f"Max    : {values.max():.4f}")


# ============================================================
# AVERAGE FUSION WEIGHT
# ============================================================

mean_mfcc = df["mfcc_weight"].mean()
mean_mel = df["mel_weight"].mean()
mean_chroma = df["chroma_weight"].mean()


print("\n")
print("=" * 70)
print("AVERAGE FEATURE CONTRIBUTION")
print("=" * 70)

print(f"MFCC     : {mean_mfcc:.4%}")

print(f"Mel      : {mean_mel:.4%}")

print(f"Chroma   : {mean_chroma:.4%}")


# ============================================================
# DOMINANT FEATURE
# ============================================================

feature_columns = [
    "mfcc_weight",
    "mel_weight",
    "chroma_weight",
]

dominant_counts = {
    "MFCC": 0,
    "Mel": 0,
    "Chroma": 0,
}


for _, row in df.iterrows():

    values = [
        row["mfcc_weight"],
        row["mel_weight"],
        row["chroma_weight"],
    ]

    dominant_index = np.argmax(values)

    if dominant_index == 0:
        dominant_counts["MFCC"] += 1

    elif dominant_index == 1:
        dominant_counts["Mel"] += 1

    else:
        dominant_counts["Chroma"] += 1


print("\n")
print("=" * 70)
print("DOMINANT FEATURE COUNT")
print("=" * 70)

total = len(df)

for feature, count in dominant_counts.items():

    print(f"{feature:8s}: " f"{count:3d}/{total} " f"({count / total:.2%})")


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "results/fusion_weight_analysis.csv"

os.makedirs(
    "results",
    exist_ok=True,
)

df.to_csv(
    output_path,
    index=False,
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Saved:",
    output_path,
)
