import os
import time
import torch

from multibranch_model import PediaLungXAI

import config as cfg
from config import EXPERIMENT_NAME

SAVE_DIR = os.path.join("results", EXPERIMENT_NAME)
MODEL_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cpu")

##############################################
# Load model
##############################################

print("Loading model...")


model = PediaLungXAI(num_classes=7, **cfg.MODEL_CONFIG["proposed"])

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

print("Model loaded successfully.")

##############################################
# Dummy Inputs
##############################################

mfcc = torch.randn(1, 1, 40, 94)

mel = torch.randn(1, 1, 128, 259)

chroma = torch.randn(1, 1, 12, 259)

##############################################
# Export ONNX
##############################################

os.makedirs("deployment", exist_ok=True)

onnx_path = "deployment/pedialung_xai.onnx"

##############################################
# Export Models
##############################################

os.makedirs("deployment", exist_ok=True)

onnx_path = "deployment/pedialung_xai.onnx"

torchscript_path = "deployment/pedialung_xai.pt"

with torch.no_grad():

    print("\nExporting ONNX...")

    torch.onnx.export(
        model,
        (mfcc, mel, chroma),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mfcc", "mel", "chroma"],
        output_names=[
            "prediction",
            "fusion_weights",
            "attention_map",
            "mel_feature_map",
        ],
    )

    print("ONNX exported.")

    print("\nExporting TorchScript...")

    scripted_model = torch.jit.trace(
        model,
        (mfcc, mel, chroma),
    )

    scripted_model.save(torchscript_path)

    print("TorchScript exported.")

##############################################
# Model Size
##############################################

onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

torchscript_size = os.path.getsize(torchscript_path) / (1024 * 1024)

print(f"ONNX Model Size       : {onnx_size:.2f} MB")
print(f"TorchScript Size      : {torchscript_size:.2f} MB")


##############################################
# Parameters
##############################################

params = sum(p.numel() for p in model.parameters())

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters     : {params:,}")

print(f"Trainable Parameters : {trainable:,}")

print("Warming up...")

with torch.no_grad():
    for _ in range(10):
        model(mfcc, mel, chroma)

##############################################
# CPU Inference Benchmark
##############################################

runs = 100

start = time.time()

with torch.no_grad():

    for _ in range(runs):

        model(mfcc, mel, chroma)

elapsed = time.time() - start

avg_time = elapsed / runs

print(f"\nAverage CPU Inference : {avg_time*1000:.2f} ms")

##############################################
# Save Deployment Report
##############################################

with open("deployment/deployment_report.txt", "w") as f:

    f.write("PediaLung-XAI Deployment Report\n\n")

    f.write(f"ONNX Model Size : {onnx_size:.2f} MB\n")
    f.write(f"TorchScript Size : {torchscript_size:.2f} MB\n")

    f.write(f"Total Parameters : {params:,}\n")

    f.write(f"Trainable Parameters : {trainable:,}\n")

    f.write(f"Average CPU Inference : {avg_time*1000:.2f} ms\n")

print("\nDeployment report saved.")
