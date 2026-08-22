import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ==========================================================
# CPU / MEMORY SETTINGS
# ==========================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORTS
# ==========================================================

from proposed_model.multifeature_dataset import MultiFeatureDataset

from proposed_model.multibranch_model import PediaLungXAI

from proposed_model.comparison_models.train_resnet18 import (
    ResNet18,
    convert_features as resnet_convert_features,
)

from proposed_model.comparison_models.train_efficientnet import (
    EfficientNetB0Model,
    convert_features as efficientnet_convert_features,
)

# ==========================================================
# PATHS
# ==========================================================

FEATURE_ROOT = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "features",
)

LABEL_CSV = os.path.join(
    FEATURE_ROOT,
    "labels.csv",
)


PROPOSED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "saved_models",
    "proposed_focal_sampler_best.pth",
)


RESNET_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "comparison_models",
    "saved_models",
    "resnet18_best.pth",
)


EFFICIENTNET_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "comparison_models",
    "saved_models",
    "efficientnet_b0_best.pth",
)


OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "proposed_model",
    "results",
    "project_model_benchmark.csv",
)


# ==========================================================
# CONFIGURATION
# ==========================================================

BATCH_SIZE = 8

WARMUP_BATCHES = 10

BENCHMARK_RUNS = 3

DEVICE = torch.device("cpu")


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("PEDIALUNG-XAI PROJECT MODEL BENCHMARK")
print("=" * 70)

print("Device:", DEVICE)


# ==========================================================
# LOAD LABELS
# ==========================================================

df = pd.read_csv(LABEL_CSV)

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

num_classes = len(label_encoder.classes_)


# ==========================================================
# EXACT SAME TEST SPLIT
# ==========================================================

trainval_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_encoded"],
)

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.125,
    random_state=42,
    stratify=trainval_df["label_encoded"],
)


print("\nDataset:")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))


# ==========================================================
# TEST DATASET
# ==========================================================

test_dataset = MultiFeatureDataset(
    dataframe=test_df,
    feature_root=FEATURE_ROOT,
    train=False,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# MODEL SIZE
# ==========================================================


def get_model_size(path):

    size_bytes = os.path.getsize(path)

    size_mb = size_bytes / 1_000_000

    return size_bytes, size_mb


# ==========================================================
# PARAMETER COUNT
# ==========================================================


def get_parameters(model):

    return sum(p.numel() for p in model.parameters())


# ==========================================================
# PROPOSED MODEL
# ==========================================================

print("\nLoading PediaLung-XAI...")

proposed_model = PediaLungXAI(
    num_classes=num_classes,
    residual=True,
    se=True,
    fusion=True,
    attention=True,
).to(DEVICE)


proposed_state = torch.load(
    PROPOSED_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

proposed_model.load_state_dict(proposed_state)

proposed_model.eval()

print("PediaLung-XAI loaded.")


# ==========================================================
# RESNET-18
# ==========================================================

print("\nLoading ResNet-18...")

resnet_model = ResNet18(num_classes=num_classes).to(DEVICE)


resnet_state = torch.load(
    RESNET_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

resnet_model.load_state_dict(resnet_state)

resnet_model.eval()

print("ResNet-18 loaded.")


# ==========================================================
# EFFICIENTNET-B0
# ==========================================================

print("\nLoading EfficientNet-B0...")

efficientnet_model = EfficientNetB0Model(num_classes=num_classes).to(DEVICE)


efficientnet_state = torch.load(
    EFFICIENTNET_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

efficientnet_model.load_state_dict(efficientnet_state)

efficientnet_model.eval()

print("EfficientNet-B0 loaded.")


# ==========================================================
# BENCHMARK FUNCTION
# ==========================================================


def benchmark_model(
    model,
    feature_converter,
    model_name,
):

    print("\n" + "=" * 70)

    print(f"BENCHMARKING: {model_name}")

    print("=" * 70)

    total_samples = 0

    total_time = 0.0

    # ------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------

    print(f"\nWarm-up batches: {WARMUP_BATCHES}")

    with torch.no_grad():

        for batch_index, (
            mfcc,
            mel,
            chroma,
            labels,
        ) in enumerate(test_loader):

            if batch_index >= WARMUP_BATCHES:
                break

            mfcc = mfcc.to(DEVICE)
            mel = mel.to(DEVICE)
            chroma = chroma.to(DEVICE)

            inputs = feature_converter(
                mfcc,
                mel,
                chroma,
            )

            _ = model(inputs)

    # ------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------

    print(f"Benchmark runs: {BENCHMARK_RUNS}")

    run_times = []

    for run in range(BENCHMARK_RUNS):

        start_time = time.perf_counter()

        samples = 0

        with torch.no_grad():

            for (
                mfcc,
                mel,
                chroma,
                labels,
            ) in test_loader:

                mfcc = mfcc.to(DEVICE)
                mel = mel.to(DEVICE)
                chroma = chroma.to(DEVICE)

                inputs = feature_converter(
                    mfcc,
                    mel,
                    chroma,
                )

                _ = model(inputs)

                samples += labels.size(0)

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        run_times.append(elapsed)

        print(f"Run {run + 1}: " f"{elapsed:.4f} seconds")

    # ------------------------------------------------------
    # Average
    # ------------------------------------------------------

    average_time = np.mean(run_times)

    average_per_sample = average_time / len(test_dataset)

    average_per_sample_ms = average_per_sample * 1000

    print("\nResults:")

    print(f"Average total time : " f"{average_time:.4f} seconds")

    print(f"Average per sample : " f"{average_per_sample_ms:.4f} ms")

    return {
        "Model": model_name,
        "Parameters": get_parameters(model),
        "Model Size (MB)": get_model_size(
            {
                "PediaLung-XAI": PROPOSED_MODEL_PATH,
                "ResNet-18": RESNET_MODEL_PATH,
                "EfficientNet-B0": EFFICIENTNET_MODEL_PATH,
            }[model_name]
        )[1],
        "Test Samples": len(test_dataset),
        "Average Inference Time (ms/sample)": average_per_sample_ms,
        "Average Total Inference Time (s)": average_time,
    }


# ==========================================================
# RUN BENCHMARK
# ==========================================================


# ----------------------------------------------------------
# PediaLungXAI accepts 3 separate features.
# ResNet/EfficientNet accept converted inputs.
# ----------------------------------------------------------


def benchmark_proposed():

    print("\n" + "=" * 70)
    print("BENCHMARKING: PediaLung-XAI")
    print("=" * 70)

    run_times = []

    for run in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        with torch.no_grad():

            for (
                mfcc,
                mel,
                chroma,
                labels,
            ) in test_loader:

                mfcc = mfcc.to(DEVICE)
                mel = mel.to(DEVICE)
                chroma = chroma.to(DEVICE)

                _ = proposed_model(
                    mfcc,
                    mel,
                    chroma,
                )

        elapsed = time.perf_counter() - start

        run_times.append(elapsed)

        print(f"Run {run + 1}: " f"{elapsed:.4f} seconds")

    avg = np.mean(run_times)

    per_sample = (avg / len(test_dataset)) * 1000

    print(f"\nAverage total time: " f"{avg:.4f} seconds")

    print(f"Average per sample: " f"{per_sample:.4f} ms")

    return {
        "Model": "PediaLung-XAI",
        "Parameters": get_parameters(proposed_model),
        "Model Size (MB)": get_model_size(PROPOSED_MODEL_PATH)[1],
        "Test Samples": len(test_dataset),
        "Average Inference Time (ms/sample)": per_sample,
        "Average Total Inference Time (s)": avg,
    }


def benchmark_resnet():

    print("\n" + "=" * 70)
    print("BENCHMARKING: ResNet-18")
    print("=" * 70)

    run_times = []

    for run in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        with torch.no_grad():

            for (
                mfcc,
                mel,
                chroma,
                labels,
            ) in test_loader:

                mfcc = mfcc.to(DEVICE)
                mel = mel.to(DEVICE)
                chroma = chroma.to(DEVICE)

                inputs = resnet_convert_features(
                    mfcc,
                    mel,
                    chroma,
                )

                _ = resnet_model(inputs)

        elapsed = time.perf_counter() - start

        run_times.append(elapsed)

        print(f"Run {run + 1}: " f"{elapsed:.4f} seconds")

    avg = np.mean(run_times)

    per_sample = (avg / len(test_dataset)) * 1000

    print(f"\nAverage total time: " f"{avg:.4f} seconds")

    print(f"Average per sample: " f"{per_sample:.4f} ms")

    return {
        "Model": "ResNet-18",
        "Parameters": get_parameters(resnet_model),
        "Model Size (MB)": get_model_size(RESNET_MODEL_PATH)[1],
        "Test Samples": len(test_dataset),
        "Average Inference Time (ms/sample)": per_sample,
        "Average Total Inference Time (s)": avg,
    }


def benchmark_efficientnet():

    print("\n" + "=" * 70)
    print("BENCHMARKING: EfficientNet-B0")
    print("=" * 70)

    run_times = []

    for run in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        with torch.no_grad():

            for (
                mfcc,
                mel,
                chroma,
                labels,
            ) in test_loader:

                mfcc = mfcc.to(DEVICE)
                mel = mel.to(DEVICE)
                chroma = chroma.to(DEVICE)

                inputs = efficientnet_convert_features(
                    mfcc,
                    mel,
                    chroma,
                )

                _ = efficientnet_model(inputs)

        elapsed = time.perf_counter() - start

        run_times.append(elapsed)

        print(f"Run {run + 1}: " f"{elapsed:.4f} seconds")

    avg = np.mean(run_times)

    per_sample = (avg / len(test_dataset)) * 1000

    print(f"\nAverage total time: " f"{avg:.4f} seconds")

    print(f"Average per sample: " f"{per_sample:.4f} ms")

    return {
        "Model": "EfficientNet-B0",
        "Parameters": get_parameters(efficientnet_model),
        "Model Size (MB)": get_model_size(EFFICIENTNET_MODEL_PATH)[1],
        "Test Samples": len(test_dataset),
        "Average Inference Time (ms/sample)": per_sample,
        "Average Total Inference Time (s)": avg,
    }


# ==========================================================
# FINAL BENCHMARK
# ==========================================================

benchmark_results = []

benchmark_results.append(benchmark_proposed())

benchmark_results.append(benchmark_resnet())

benchmark_results.append(benchmark_efficientnet())


# ==========================================================
# SAVE
# ==========================================================

benchmark_df = pd.DataFrame(benchmark_results)


benchmark_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ==========================================================
# DISPLAY
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL PROJECT MODEL BENCHMARK")
print("=" * 70)

print(benchmark_df.to_string(index=False))


print("\nSaved to:")

print(OUTPUT_PATH)


print("\n")
print("=" * 70)
print("BENCHMARK COMPLETED")
print("=" * 70)
