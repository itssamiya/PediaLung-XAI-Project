import os
import time
import numpy as np
import torch
import onnxruntime as ort

from multibranch_model import PediaLungXAI
from config import MODEL_CONFIG
from config import EXPERIMENT_NAME
from config import MODEL_DIR

# ==========================================================
# Configuration
# ==========================================================

device = torch.device("cpu")

ONNX_PATH = "deployment/pedialung_xai.onnx"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_best.pth")


# ==========================================================
# Check ONNX file
# ==========================================================

print("=" * 60)
print("ONNX Deployment Test")
print("=" * 60)

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

print("ONNX model found.")
print("Path:", ONNX_PATH)


# ==========================================================
# Create test inputs
# ==========================================================

mfcc = torch.randn(1, 1, 40, 94, dtype=torch.float32)

mel = torch.randn(1, 1, 128, 259, dtype=torch.float32)

chroma = torch.randn(1, 1, 12, 259, dtype=torch.float32)


# ==========================================================
# Load PyTorch model
# ==========================================================

print("\nLoading PyTorch model...")

model = PediaLungXAI(num_classes=7, **MODEL_CONFIG["proposed"]).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

model.eval()

print("PyTorch model loaded.")


# ==========================================================
# PyTorch inference
# ==========================================================

with torch.no_grad():

    pytorch_output, _, _, _ = model(mfcc, mel, chroma)

pytorch_prediction = torch.softmax(pytorch_output, dim=1).numpy()


# ==========================================================
# Load ONNX model
# ==========================================================

print("\nLoading ONNX model...")

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

print("ONNX model loaded.")

print("\nONNX Inputs:")

for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nONNX Outputs:")

for out in session.get_outputs():
    print(out.name, out.shape, out.type)


# ==========================================================
# ONNX inference
# ==========================================================

onnx_inputs = {
    "mfcc": mfcc.numpy(),
    "mel": mel.numpy(),
    "chroma": chroma.numpy(),
}


start = time.perf_counter()

onnx_outputs = session.run(None, onnx_inputs)

elapsed = (time.perf_counter() - start) * 1000


# ==========================================================
# Compare prediction
# ==========================================================

onnx_prediction = onnx_outputs[0]

# Convert logits to probabilities
onnx_prediction = torch.softmax(torch.tensor(onnx_prediction), dim=1).numpy()


pytorch_class = np.argmax(pytorch_prediction, axis=1)[0]

onnx_class = np.argmax(onnx_prediction, axis=1)[0]


print("\n" + "=" * 60)
print("Prediction Comparison")
print("=" * 60)

print("PyTorch predicted class :", pytorch_class)
print("ONNX predicted class    :", onnx_class)

if pytorch_class == onnx_class:
    print("Prediction Match        : YES")
else:
    print("Prediction Match        : NO")


# ==========================================================
# Numerical difference
# ==========================================================

difference = np.max(np.abs(pytorch_prediction - onnx_prediction))

print(f"Maximum probability difference : " f"{difference:.8f}")


# ==========================================================
# ONNX CPU benchmark
# ==========================================================

print("\nRunning ONNX CPU benchmark...")

warmup = 10
runs = 100


for _ in range(warmup):

    session.run(None, onnx_inputs)


start = time.perf_counter()

for _ in range(runs):

    session.run(None, onnx_inputs)

elapsed = time.perf_counter() - start

average_time = (elapsed / runs) * 1000


print(f"Average ONNX CPU inference : " f"{average_time:.2f} ms")


# ==========================================================
# Save deployment verification
# ==========================================================

os.makedirs("deployment", exist_ok=True)

with open("deployment/onnx_verification.txt", "w") as f:

    f.write("PediaLung-XAI ONNX Deployment Verification\n\n")

    f.write(f"PyTorch predicted class : " f"{pytorch_class}\n")

    f.write(f"ONNX predicted class : " f"{onnx_class}\n")

    f.write(f"Prediction match : " f"{pytorch_class == onnx_class}\n")

    f.write(f"Maximum probability difference : " f"{difference:.8f}\n")

    f.write(f"Average ONNX CPU inference : " f"{average_time:.2f} ms\n")

print("\nVerification report saved:")
print("deployment/onnx_verification.txt")

print("\n" + "=" * 60)
print("ONNX deployment test completed.")
print("=" * 60)
