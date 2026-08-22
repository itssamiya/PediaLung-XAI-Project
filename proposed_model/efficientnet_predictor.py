import gc
import json
import os
import time

# Set BLAS thread limits BEFORE importing numerical/deep learning libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torchvision.models import efficientnet_b0

from utils.preprocessing import (
    extract_chroma,
    extract_mel,
    extract_mfcc,
    load_audio,
    normalize_length,
    wavelet_denoise,
)

# Limit PyTorch CPU thread allocation
torch.set_num_threads(2)


class EfficientNetGradCAM:

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # Last convolutional feature block
        self.target_layer = self.model.backbone.features[-1]

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, inputs, target_class):
        self.model.zero_grad()

        # Ensure input tracks gradients for backward pass
        inputs = inputs.requires_grad_(True)
        outputs = self.model(inputs)

        target = outputs[:, target_class]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Safely squeeze batch and channel dimensions (B=1, C=1, H, W -> H, W)
        return cam.squeeze(0).squeeze(0).cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEATURE_ROOT = os.path.join(PROJECT_ROOT, "features")
LABEL_CSV = os.path.join(FEATURE_ROOT, "labels.csv")

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "comparison_models", "saved_models", "efficientnet_b0_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TARGET FEATURE SIZE
# ============================================================

TARGET_HEIGHT = 128
TARGET_WIDTH = 259

NUM_CLASSES = 7


# ============================================================
# EFFICIENTNET-B0 MODEL
# ============================================================


class EfficientNetB0Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# FEATURE CONVERSION
# EXACTLY MATCHES TRAINING
# ============================================================


def convert_features(mfcc, mel, chroma):
    mfcc = F.interpolate(
        mfcc,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    mel = F.interpolate(
        mel,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    chroma = F.interpolate(
        chroma,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    x = torch.cat([mfcc, mel, chroma], dim=1)
    return x


# ============================================================
# PREDICTOR
# ============================================================


class EfficientNetPredictor:

    def __init__(self):
        print("=" * 60)
        print("INITIALIZING EFFICIENTNET-B0 PREDICTOR")
        print("=" * 60)

        self.device = DEVICE
        print("Device:", self.device)
        print("Model:", MODEL_PATH)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"EfficientNet model not found:\n{MODEL_PATH}")

        if not os.path.exists(LABEL_CSV):
            raise FileNotFoundError(f"Label file not found:\n{LABEL_CSV}")

        # ----------------------------------------------------
        # Load labels
        # ----------------------------------------------------
        df = pd.read_csv(LABEL_CSV)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df["label"])
        self.class_names = self.label_encoder.classes_

        print("\nClasses:")
        for i, name in enumerate(self.class_names):
            print(f"{i}: {name}")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------
        print("\nLoading EfficientNet-B0...")
        self.model = EfficientNetB0Model(num_classes=len(self.class_names)).to(
            self.device
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        print("EfficientNet-B0 loaded successfully.")
        print("Total Parameters:", f"{self.total_params:,}")
        print("=" * 60)

        self.gradcam = EfficientNetGradCAM(self.model)

    def generate_gradcam(self, inputs, predicted_class):

        # 1. Generate GradCAM heatmap from backprop gradients
        with torch.set_grad_enabled(True):
            heatmap = self.gradcam.generate(inputs, predicted_class)

        # 2. Extract Mel-Spectrogram directly from Channel 1 [1, 3, 128, 259]
        mel_tensor = inputs[0, 1].cpu().detach().numpy()  # Shape: (128, 259)

        # 3. Robust percentile normalization
        vmin, vmax = np.percentile(mel_tensor, 1), np.percentile(mel_tensor, 99)
        mel_clipped = np.clip(mel_tensor, vmin, vmax)
        mel_norm = (mel_clipped - vmin) / (vmax - vmin + 1e-8)

        # 4. Resize and normalize heatmap to match input matrix dimensions
        target_h, target_w = mel_norm.shape
        heatmap_resized = cv2.resize(
            heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
            heatmap_resized.max() - heatmap_resized.min() + 1e-8
        )

        # 5. Build matplotlib side-by-side plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

        # Upper Plot: Base Mel Spectrogram
        im0 = axes[0].imshow(
            mel_norm,
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        axes[0].set_title(
            "Base Mel-Spectrogram (Input Channel 1)",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_ylabel("Frequency Bins")

        # Lower Plot: GradCAM Overlay
        axes[1].imshow(
            mel_norm,
            aspect="auto",
            origin="lower",
            cmap="gray",
            alpha=0.6,
        )
        axes[1].imshow(
            heatmap_norm,
            aspect="auto",
            origin="lower",
            cmap="jet",
            alpha=0.5,
        )
        axes[1].set_title(
            f"GradCAM Attention Overlay (Class: {predicted_class})",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Frequency Bins")

        plt.tight_layout()

        # 6. Save high-resolution visualization
        os.makedirs("results/gradcam", exist_ok=True)
        output_path = os.path.join(
            "results", "gradcam", "latest_efficientnet_gradcam.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    # ========================================================
    # AUDIO PREPROCESSING
    # ========================================================

    def preprocess(self, wav_path):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found:\n{wav_path}")

        print("\nLoading audio...")
        signal, sr = load_audio(wav_path)
        print("Audio loaded.")

        signal = wavelet_denoise(signal)
        signal = normalize_length(signal, sr)

        mfcc = extract_mfcc(signal, sr)
        mel = extract_mel(signal, sr)
        chroma = extract_chroma(signal, sr)

        # Clean memory explicitly after feature extraction
        del signal
        gc.collect()

        return mfcc, mel, chroma

    # ========================================================
    # PREPARE TENSORS
    # ========================================================

    def prepare_features(self, mfcc, mel, chroma):
        mfcc_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        chroma_t = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mfcc_t = mfcc_t.to(self.device)
        mel_t = mel_t.to(self.device)
        chroma_t = chroma_t.to(self.device)

        inputs = convert_features(mfcc_t, mel_t, chroma_t)
        return inputs

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, wav_path):
        total_start = time.perf_counter()

        preprocessing_start = time.perf_counter()
        mfcc, mel, chroma = self.preprocess(wav_path)
        inputs = self.prepare_features(mfcc, mel, chroma)
        preprocessing_time = (time.perf_counter() - preprocessing_start) * 1000

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
        inference_time = (time.perf_counter() - inference_start) * 1000

        confidence, predicted = torch.max(probabilities, dim=1)
        predicted_index = predicted.item()
        prediction = self.class_names[predicted_index]

        # Generate and save GradCAM visualization
        gradcam_path = self.generate_gradcam(inputs, predicted_index)

        k = min(3, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

        top_predictions = []
        for probability, index in zip(top_probs[0], top_indices[0]):
            class_name = self.class_names[index.item()]
            top_predictions.append((class_name, float(probability.item())))

        total_time = (time.perf_counter() - total_start) * 1000

        result = {
            "prediction": prediction,
            "predicted_class_index": predicted_index,
            "confidence": float(confidence.item()),
            "top_predictions": top_predictions,
            "preprocessing_time_ms": preprocessing_time,
            "inference_time_ms": inference_time,
            "total_time_ms": total_time,
            "model": "EfficientNet-B0",
            "parameters": self.total_params,
            "gradcam_path": gradcam_path,
        }

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    predictor = EfficientNetPredictor()

    print("\nEnter path to a WAV file.")
    wav_path = input("WAV path: ").strip().strip('"')

    result = predictor.predict(wav_path)

    print("\n")
    print("=" * 60)
    print("EFFICIENTNET-B0 PREDICTION")
    print("=" * 60)

    print("\nPrediction:", result["prediction"])
    print("Confidence:", f"{result['confidence'] * 100:.2f}%")

    print("\nTop-3 Predictions:")
    for name, probability in result["top_predictions"]:
        print(f"{name:20s}{probability * 100:.2f}%")

    print("\nPerformance:")
    print("Preprocessing:", f"{result['preprocessing_time_ms']:.2f} ms")
    print("Inference:", f"{result['inference_time_ms']:.2f} ms")
    print("Total:", f"{result['total_time_ms']:.2f} ms")

    print("\nParameters:", f"{result['parameters']:,}")
    print("=" * 60)

    print("\nGradCAM Saved To:", result["gradcam_path"])
