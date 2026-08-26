import os

# Limit BLAS threads (helps on low-memory systems)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import csv
import gc
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm
from skimage.restoration import denoise_wavelet

# ==========================================================
# Output Folder
# ==========================================================

FEATURE_DIR = Path("features/chroma")
if not FEATURE_DIR.exists():
    FEATURE_DIR.mkdir(parents=True)



DATASET_PATH = Path(r"D:\Research\SPRSound-main\SPRSound-main\BioCAS2022")

WAV_FOLDER = DATASET_PATH / "train2022_wav"
JSON_FOLDER = DATASET_PATH / "train2022_json"

# ==========================================================

N_CHROMA = 12


def make_fixed_length(signal, sr, target_seconds=6):
    """
    Repeat short signals until they become 6 seconds.
    Trim long signals to exactly 6 seconds.
    """

    target_samples = sr * target_seconds

    while len(signal) < target_samples:
        signal = np.concatenate((signal, signal))

    signal = signal[:target_samples]

    return signal



feature_index = 0


# ==========================================================
# Load Dataset
# ==========================================================

json_files = sorted(JSON_FOLDER.glob("*.json"))

print(f"Found {len(json_files)} recordings.\n")

# ==========================================================
# Process Dataset
# ==========================================================

for json_file in tqdm(json_files):

    wav_file = WAV_FOLDER / (json_file.stem + ".wav")

    if not wav_file.exists():
        continue

    with open(json_file, "r") as f:
        annotation = json.load(f)

    signal, sr = librosa.load(wav_file, sr=None)

    events = annotation.get("event_annotation", [])

    if len(events) == 0:
        del signal
        gc.collect()
        continue

    for event in events:

        label = event["type"]

        if label.strip() == "":
            continue

        start_ms = int(event["start"])
        end_ms = int(event["end"])

        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)

        event_signal = signal[start_sample:end_sample]

        if len(event_signal) == 0:
            continue

        # ----------------------------------------
        # Wavelet Denoising
        # ----------------------------------------

        event_signal = denoise_wavelet(
            event_signal,
            method="BayesShrink",
            mode="soft",
            wavelet="sym8",
            wavelet_levels=3,
            rescale_sigma=True
        )

        # ----------------------------------------
        # Make Fixed Length
        # ----------------------------------------

        event_signal = make_fixed_length(event_signal, sr)

       # ----------------------------------------
        # ----------------------------------------
        # Chromagram Extraction
        # ----------------------------------------

        chroma = librosa.feature.chroma_stft(
            y=event_signal,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_chroma=N_CHROMA
        )

        # ----------------------------------------
        # Save Feature
        # ----------------------------------------

        feature_name = f"{feature_index:06d}.npy"

        np.save(
            FEATURE_DIR / feature_name,
            chroma.astype(np.float32)
        )

        feature_index += 1

        del chroma
        del event_signal

        gc.collect()

    # Free recording memory
    del signal
    gc.collect()

# ==========================================================
# Finish
# ==========================================================


print("\n===================================")
print("Chromagram extraction completed!")
print("===================================")
print(f"Total respiratory events : {feature_index}")
print("Chroma folder : features/chroma")
print("Using existing labels : features/labels.csv")
