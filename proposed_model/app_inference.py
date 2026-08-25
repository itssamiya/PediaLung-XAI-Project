import gc
import os
import sys
import time

# CPU Performance constraints to prevent memory leaks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Force path resolution so imports work regardless of working directory
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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

torch.set_num_threads(1)

from utils.preprocessing import (
    extract_chroma,
    extract_mel,
    extract_mfcc,
    load_audio,
    normalize_length,
    wavelet_denoise,
)

TARGET_HEIGHT = 128
TARGET_WIDTH = 259


class EfficientNetGradCAM:

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

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
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = self.model(inputs)

        target = outputs[:, target_class]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam_np = cam.squeeze().cpu().numpy()

        # Clean gradient references
        self.model.zero_grad()
        del outputs, target, gradients, activations, weights, cam
        return cam_np


class EfficientNetB0Model(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class PediaLungAppPredictor:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using Device:", self.device)

        # Locate files flexibly
        label_csv = os.path.join(ROOT_DIR, "features", "labels.csv")
        if not os.path.exists(label_csv):
            label_csv = os.path.join(FILE_DIR, "features", "labels.csv")

        model_path = os.path.join(
            FILE_DIR, "comparison_models", "saved_models", "efficientnet_b0_best.pth"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                ROOT_DIR,
                "proposed_model",
                "comparison_models",
                "saved_models",
                "efficientnet_b0_best.pth",
            )

        if not os.path.exists(label_csv):
            raise FileNotFoundError(f"Label file not found: {label_csv}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"EfficientNet model not found: {model_path}")

        # Load labels
        df = pd.read_csv(label_csv)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df["label"])
        self.class_names = self.label_encoder.classes_

        # Load model
        print("Loading EfficientNet-B0 Model...")
        self.model = EfficientNetB0Model(num_classes=len(self.class_names)).to(
            self.device
        )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.gradcam = EfficientNetGradCAM(self.model)
        print("EfficientNet-B0 Predictor and Grad-CAM ready.")

    def preprocess(self, wav_path):
        signal, sr = load_audio(wav_path)
        signal = wavelet_denoise(signal)
        signal = normalize_length(signal, sr)

        mfcc = extract_mfcc(signal, sr)
        mel = extract_mel(signal, sr)
        chroma = extract_chroma(signal, sr)

        del signal
        gc.collect()

        return mfcc, mel, chroma

    def prepare_tensors(self, mfcc, mel, chroma):
        mfcc_t = (
            torch.tensor(mfcc, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        mel_t = (
            torch.tensor(mel, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        chroma_t = (
            torch.tensor(chroma, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        mfcc_t = F.interpolate(
            mfcc_t,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        mel_t = F.interpolate(
            mel_t,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        chroma_t = F.interpolate(
            chroma_t,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

        return torch.cat([mfcc_t, mel_t, chroma_t], dim=1)

    def create_gradcam(self, inputs, predicted_index, save_path):
        # Generate Grad-CAM heatmap using class index integer
        heatmap = self.gradcam.generate(inputs, predicted_index)

        mel_tensor = inputs[0, 1].cpu().detach().numpy()
        vmin, vmax = np.percentile(mel_tensor, 1), np.percentile(mel_tensor, 99)
        mel_clipped = np.clip(mel_tensor, vmin, vmax)
        mel_norm = (mel_clipped - vmin) / (vmax - vmin + 1e-8)

        target_h, target_w = mel_norm.shape
        heatmap_resized = cv2.resize(
            heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
            heatmap_resized.max() - heatmap_resized.min() + 1e-8
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

        axes[0].imshow(mel_norm, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title(
            "Base Mel-Spectrogram (Input Channel 1)",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_ylabel("Frequency Bins")

        axes[1].imshow(mel_norm, aspect="auto", origin="lower", cmap="gray", alpha=0.6)
        axes[1].imshow(
            heatmap_norm, aspect="auto", origin="lower", cmap="jet", alpha=0.5
        )
        axes[1].set_title(
            f"GradCAM Attention Overlay (Class: {self.class_names[predicted_index]})",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Frequency Bins")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Force garbage collection to prevent CPU memory accumulation
        del heatmap, heatmap_resized, mel_norm, mel_tensor
        gc.collect()

        return save_path

    def predict(self, wav_path):
        total_start = time.perf_counter()

        pre_start = time.perf_counter()
        mfcc, mel, chroma = self.preprocess(wav_path)
        inputs = self.prepare_tensors(mfcc, mel, chroma)
        pre_time = (time.perf_counter() - pre_start) * 1000

        infer_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
        infer_time = (time.perf_counter() - infer_start) * 1000

        confidence, predicted = torch.max(probabilities, dim=1)
        predicted_index = predicted.item()
        prediction = self.class_names[predicted_index]

        k = min(3, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

        top3 = [
            (self.class_names[idx.item()], float(prob.item()))
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        grad_start = time.perf_counter()
        save_path = os.path.join(ROOT_DIR, "results", "app_gradcam.png")
        self.create_gradcam(inputs, predicted_index, save_path)
        grad_time = (time.perf_counter() - grad_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # Cleanup tensor memory
        del inputs, outputs, probabilities
        gc.collect()

        return {
            "prediction": prediction,
            "confidence": float(confidence.item()),
            "binary_result": "Normal" if prediction == "Normal" else "Abnormal",
            "top3": top3,
            "gradcam_path": save_path,
            "preprocessing_time": pre_time,
            "inference_time": infer_time,
            "gradcam_time": grad_time,
            "total_time": total_time,
        }
