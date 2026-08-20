import os
import time
import torch
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from multibranch_model import PediaLungXAI

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

BINARY_MODEL_PATH = os.path.join("saved_models", "hierarchical_binary_best.pth")

ABNORMAL_MODEL_PATH = os.path.join(
    "saved_models", "hierarchical_abnormal_ce_sampler_best.pth"
)

NUM_CLASSES_BINARY = 2
NUM_CLASSES_ABNORMAL = 6


# ============================================================
# MODEL INFORMATION
# ============================================================


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024**2)


def benchmark_model(model, input_shapes, runs=50, warmup=10):
    """
    Measures CPU inference time.

    input_shapes:
        list of tuples representing tensor shapes.
        Example:
        [(1, 1, 40, 100), (1, 1, 128, 100), (1, 1, 12, 100)]
    """

    model.eval()

    inputs = [torch.randn(*shape, device=DEVICE) for shape in input_shapes]

    with torch.no_grad():

        # Warm-up
        for _ in range(warmup):
            model(*inputs)

        start = time.perf_counter()

        for _ in range(runs):
            model(*inputs)

        end = time.perf_counter()

    avg_time = (end - start) / runs

    return avg_time * 1000


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("PediaLung-XAI MODEL EFFICIENCY EVALUATION")
print("=" * 60)

print("Device:", DEVICE)


# ------------------------------------------------------------
# Binary model
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BINARY MODEL")
print("=" * 60)

binary_model = PediaLungXAI(num_classes=NUM_CLASSES_BINARY).to(DEVICE)

binary_checkpoint = torch.load(BINARY_MODEL_PATH, map_location=DEVICE)

if "model_state_dict" in binary_checkpoint:
    binary_model.load_state_dict(binary_checkpoint["model_state_dict"])
else:
    binary_model.load_state_dict(binary_checkpoint)

binary_params, binary_trainable = get_model_info(binary_model)

print("Total Parameters :", f"{binary_params:,}")
print("Trainable Params :", f"{binary_trainable:,}")

print("Checkpoint Size  :", f"{get_file_size_mb(BINARY_MODEL_PATH):.2f} MB")


# ------------------------------------------------------------
# Abnormal model
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ABNORMAL MODEL")
print("=" * 60)

abnormal_model = PediaLungXAI(num_classes=NUM_CLASSES_ABNORMAL).to(DEVICE)

abnormal_checkpoint = torch.load(ABNORMAL_MODEL_PATH, map_location=DEVICE)

if "model_state_dict" in abnormal_checkpoint:
    abnormal_model.load_state_dict(abnormal_checkpoint["model_state_dict"])
else:
    abnormal_model.load_state_dict(abnormal_checkpoint)

abnormal_params, abnormal_trainable = get_model_info(abnormal_model)

print("Total Parameters :", f"{abnormal_params:,}")
print("Trainable Params :", f"{abnormal_trainable:,}")

print("Checkpoint Size  :", f"{get_file_size_mb(ABNORMAL_MODEL_PATH):.2f} MB")


# ------------------------------------------------------------
# Combined parameter count
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HIERARCHICAL SYSTEM")
print("=" * 60)

combined_params = binary_params + abnormal_params

print("Combined Parameters:", f"{combined_params:,}")

print(
    "Combined Checkpoint Size:",
    f"{get_file_size_mb(BINARY_MODEL_PATH) + get_file_size_mb(ABNORMAL_MODEL_PATH):.2f} MB",
)


print("\nEvaluation completed.")
