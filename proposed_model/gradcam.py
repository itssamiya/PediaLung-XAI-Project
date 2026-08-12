import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from multibranch_model import PediaLungXAI
from gradcam_plus_plus import GradCAMPlusPlus
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from multifeature_dataset import MultiFeatureDataset

from torch.utils.data import DataLoader

from config import MODEL_CONFIG
from config import EXPERIMENT_NAME
from config import SAVE_DIR
from config import MODEL_DIR

print("=" * 60)
print("Running file:", os.path.abspath(__file__))
print("=" * 60)

USE_GRADCAM_PLUS_PLUS = True


class GradCAM:

    def __init__(self, model):

        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        target_layer = self.model.mel_encoder.features[14]
        print(target_layer)

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, mfcc, mel, chroma, target_class=None):

        self.model.zero_grad()

        output, _, _, _ = self.model(mfcc, mel, chroma)

        if target_class is None:

            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        print("Gradient mean :", gradients.abs().mean().item())
        print("Activation mean :", activations.abs().mean().item())

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=activations.device)

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = F.relu(cam)

        print(cam.min().item(), cam.max().item())

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam, (mel.shape[-1], mel.shape[-2]), interpolation=cv2.INTER_CUBIC
        )

        cam = cv2.GaussianBlur(cam, (9, 9), 0)

        cam = np.maximum(cam, 0)

        cam = cam / (cam.max() + 1e-8)

        cam = np.power(cam, 0.6)

        return cam


# Visualization


def visualize_gradcam(
    mel_spec,
    heatmap,
    save_path,
    true_label,
    pred_label,
    confidence=None,
    top_predictions=None,
):

    mel_spec = mel_spec.squeeze().cpu().numpy()

    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    fig = plt.figure(figsize=(14, 8))

    #########################################################
    # Original Mel Spectrogram
    #########################################################

    plt.subplot(2, 2, 1)

    plt.imshow(
        mel_spec,
        cmap="gray",
        origin="lower",
        aspect="auto",
    )

    plt.title("Original Mel Spectrogram")

    plt.xlabel("Time")

    plt.ylabel("Mel Spectrogram")

    #########################################################
    # Heatmap
    #########################################################

    plt.subplot(2, 2, 2)

    plt.imshow(
        heatmap,
        cmap="jet",
        origin="lower",
        aspect="auto",
    )

    if USE_GRADCAM_PLUS_PLUS:
        plt.title("Grad-CAM++ Heatmap")
    else:
        plt.title("Grad-CAM Heatmap")

    plt.xlabel("Time")

    #########################################################
    # Overlay
    #########################################################

    plt.subplot(2, 2, 3)

    plt.imshow(
        mel_spec,
        cmap="gray",
        origin="lower",
        aspect="auto",
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.55,
        origin="lower",
        aspect="auto",
    )

    plt.title("Overlay")

    plt.xlabel("Time")

    #########################################################
    # Prediction Panel
    #########################################################

    plt.subplot(2, 2, 4)

    plt.axis("off")

    txt = f"Prediction : {pred_label}\n"

    if confidence is not None:

        txt += f"Confidence : {confidence*100:.2f}%\n\n"

    txt += f"Ground Truth : {true_label}\n\n"

    if top_predictions is not None:

        txt += "Top-3 Predictions\n\n"

        for cls, prob in top_predictions:

            txt += f"{cls:18s}{prob*100:.2f}%\n"

    plt.text(
        0,
        1,
        txt,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
