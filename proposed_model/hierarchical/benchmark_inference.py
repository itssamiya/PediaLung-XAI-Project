import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from multibranch_model import PediaLungXAI

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEVICE = torch.device("cpu")

BINARY_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "hierarchical_binary_best.pth"
)

ABNORMAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "hierarchical_abnormal_ce_sampler_best.pth"
)

TEST_CSV = os.path.join(
    PROJECT_ROOT, "results", "hierarchical_final", "hierarchical_probabilities.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "hierarchical_final")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------


def load_model(path, num_classes):

    model = PediaLungXAI(num_classes=num_classes).to(DEVICE)

    checkpoint = torch.load(path, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ------------------------------------------------------------
# Locate feature files
# ------------------------------------------------------------


def find_feature_file(filename):

    possible_paths = [
        os.path.join(PROJECT_ROOT, "features", "mfcc", filename),
        os.path.join(PROJECT_ROOT, "features", "mel", filename),
        os.path.join(PROJECT_ROOT, "features", "chroma", filename),
    ]

    return possible_paths


# ------------------------------------------------------------
# Load one sample
# ------------------------------------------------------------


def load_sample(filename):

    base = os.path.splitext(filename)[0]

    mfcc_path = os.path.join(PROJECT_ROOT, "features", "mfcc", base + ".npy")

    mel_path = os.path.join(PROJECT_ROOT, "features", "mel", base + ".npy")

    chroma_path = os.path.join(PROJECT_ROOT, "features", "chroma", base + ".npy")

    mfcc = np.load(mfcc_path)
    mel = np.load(mel_path)
    chroma = np.load(chroma_path)

    # Ensure channel dimension
    if mfcc.ndim == 2:
        mfcc = mfcc[np.newaxis, :]

    if mel.ndim == 2:
        mel = mel[np.newaxis, :]

    if chroma.ndim == 2:
        chroma = chroma[np.newaxis, :]

    mfcc = torch.tensor(mfcc, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    mel = torch.tensor(mel, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    chroma = torch.tensor(chroma, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    return mfcc, mel, chroma


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

print("=" * 60)
print("PediaLung-XAI INFERENCE TIME BENCHMARK")
print("=" * 60)

print("Device:", DEVICE)


binary_model = load_model(BINARY_MODEL_PATH, 2)

abnormal_model = load_model(ABNORMAL_MODEL_PATH, 6)

print("Models loaded successfully.")


df = pd.read_csv(TEST_CSV)

print("Test samples:", len(df))


# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

warmup_count = min(10, len(df))

print("\nRunning warm-up...")

with torch.no_grad():

    for i in range(warmup_count):

        try:
            mfcc, mel, chroma = load_sample(df.iloc[i]["filename"])

            binary_model(mfcc, mel, chroma)

            abnormal_model(mfcc, mel, chroma)

        except Exception as e:

            print("Warm-up sample failed:", df.iloc[i]["filename"], e)


# ------------------------------------------------------------
# Binary timing
# ------------------------------------------------------------

binary_times = []
abnormal_times = []
hierarchical_times = []

processed = 0


print("\nBenchmarking...")


with torch.no_grad():

    for _, row in df.iterrows():

        filename = row["filename"]

        try:

            mfcc, mel, chroma = load_sample(filename)

            # -------------------------
            # Binary model
            # -------------------------

            start = time.perf_counter()

            binary_model(mfcc, mel, chroma)

            binary_time = (time.perf_counter() - start) * 1000

            # -------------------------
            # Abnormal model
            # -------------------------

            start = time.perf_counter()

            abnormal_model(mfcc, mel, chroma)

            abnormal_time = (time.perf_counter() - start) * 1000

            # -------------------------
            # Hierarchical
            # -------------------------

            hierarchical_time = binary_time + abnormal_time

            binary_times.append(binary_time)
            abnormal_times.append(abnormal_time)
            hierarchical_times.append(hierarchical_time)

            processed += 1

            if processed % 100 == 0:
                print(f"Processed {processed}/{len(df)}")

        except Exception as e:

            print("Failed:", filename, e)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INFERENCE BENCHMARK RESULTS")
print("=" * 60)


def report(name, values):

    values = np.array(values)

    print(f"\n{name}")

    print("Mean inference time   :", f"{values.mean():.3f} ms")

    print("Median inference time :", f"{np.median(values):.3f} ms")

    print("Minimum               :", f"{values.min():.3f} ms")

    print("Maximum               :", f"{values.max():.3f} ms")


report("Binary Model", binary_times)

report("Abnormal Model", abnormal_times)

report("Hierarchical System", hierarchical_times)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "binary_ms": binary_times,
        "abnormal_ms": abnormal_times,
        "hierarchical_ms": hierarchical_times,
    }
)

output_csv = os.path.join(OUTPUT_DIR, "inference_benchmark.csv")

results.to_csv(output_csv, index=False)


summary_path = os.path.join(OUTPUT_DIR, "inference_benchmark.txt")

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("PediaLung-XAI Inference Benchmark\n")

    f.write("Device: CPU\n\n")

    f.write(f"Samples processed: {processed}\n\n")

    f.write(f"Binary mean: " f"{np.mean(binary_times):.3f} ms\n")

    f.write(f"Binary median: " f"{np.median(binary_times):.3f} ms\n")

    f.write(f"Abnormal mean: " f"{np.mean(abnormal_times):.3f} ms\n")

    f.write(f"Abnormal median: " f"{np.median(abnormal_times):.3f} ms\n")

    f.write(f"Hierarchical mean: " f"{np.mean(hierarchical_times):.3f} ms\n")

    f.write(f"Hierarchical median: " f"{np.median(hierarchical_times):.3f} ms\n")


print("\nResults saved to:")
print(output_csv)
print(summary_path)

print("\nBenchmark completed.")
