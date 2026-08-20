import os
import sys
import time
import numpy as np
import torch

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multibranch_model import PediaLungXAI
import config as cfg

DEVICE = torch.device("cpu")

# ============================================================
# MODEL PATHS
# ============================================================

BINARY_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "hierarchical_binary_best.pth"
)

ABNORMAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "hierarchical_abnormal_ce_sampler_best.pth"
)

# ============================================================
# CLASS DEFINITIONS
# ============================================================

ABNORMAL_CLASSES = [
    "Coarse Crackle",
    "Fine Crackle",
    "Rhonchi",
    "Stridor",
    "Wheeze",
    "Wheeze+Crackle",
]

FINAL_CLASSES = [
    "Normal",
    "Coarse Crackle",
    "Fine Crackle",
    "Rhonchi",
    "Stridor",
    "Wheeze",
    "Wheeze+Crackle",
]


class HierarchicalPredictor:

    def __init__(self):

        print("Loading PediaLung-XAI hierarchical models...")

        self.binary_model = PediaLungXAI(
            num_classes=2, **cfg.MODEL_CONFIG["proposed"]
        ).to(DEVICE)

        self.abnormal_model = PediaLungXAI(
            num_classes=6, **cfg.MODEL_CONFIG["proposed"]
        ).to(DEVICE)

        self.binary_model.load_state_dict(
            torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
        )

        self.abnormal_model.load_state_dict(
            torch.load(ABNORMAL_MODEL_PATH, map_location=DEVICE)
        )

        self.binary_model.eval()
        self.abnormal_model.eval()

        print("Binary model loaded.")
        print("Abnormal model loaded.")
        print("PediaLung-XAI ready.")

    # ========================================================
    # MODEL FORWARD
    # ========================================================

    @staticmethod
    def _forward(model, mfcc, mel, chroma):

        with torch.no_grad():

            outputs, fusion_weights, attention_map, feature_map = model(
                mfcc, mel, chroma
            )

            probabilities = torch.softmax(outputs, dim=1)

        return (
            probabilities.cpu().numpy()[0],
            fusion_weights,
            attention_map,
            feature_map,
        )

    # ========================================================
    # PREDICT FROM FEATURES
    # ========================================================

    def predict_features(self, mfcc, mel, chroma):

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Convert numpy -> tensor
        # ----------------------------------------------------

        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        mel = torch.tensor(mel, dtype=torch.float32)

        chroma = torch.tensor(chroma, dtype=torch.float32)

        # ----------------------------------------------------
        # Ensure batch/channel dimensions
        # ----------------------------------------------------

        if mfcc.ndim == 2:
            mfcc = mfcc.unsqueeze(0).unsqueeze(0)

        elif mfcc.ndim == 3:
            mfcc = mfcc.unsqueeze(0)

        if mel.ndim == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)

        elif mel.ndim == 3:
            mel = mel.unsqueeze(0)

        if chroma.ndim == 2:
            chroma = chroma.unsqueeze(0).unsqueeze(0)

        elif chroma.ndim == 3:
            chroma = chroma.unsqueeze(0)

        mfcc = mfcc.to(DEVICE)
        mel = mel.to(DEVICE)
        chroma = chroma.to(DEVICE)

        # ====================================================
        # STAGE 1: NORMAL vs ABNORMAL
        # ====================================================

        binary_probs, _, _, _ = self._forward(self.binary_model, mfcc, mel, chroma)

        normal_probability = float(binary_probs[0])
        abnormal_probability = float(binary_probs[1])

        # ====================================================
        # STAGE 2
        # ====================================================

        abnormal_probs, fusion_weights, attention_map, feature_map = self._forward(
            self.abnormal_model, mfcc, mel, chroma
        )

        # ====================================================
        # HIERARCHICAL PROBABILITY
        # ====================================================

        final_probabilities = np.zeros(len(FINAL_CLASSES), dtype=np.float64)

        # Normal
        final_probabilities[0] = normal_probability

        # Abnormal classes
        for i in range(len(ABNORMAL_CLASSES)):

            final_probabilities[i + 1] = abnormal_probability * float(abnormal_probs[i])

        # Numerical normalization
        probability_sum = final_probabilities.sum()

        if probability_sum > 0:

            final_probabilities /= probability_sum

        # ====================================================
        # FINAL PREDICTION
        # ====================================================

        prediction_index = int(np.argmax(final_probabilities))

        prediction = FINAL_CLASSES[prediction_index]

        confidence = float(final_probabilities[prediction_index])

        # ====================================================
        # TOP 3
        # ====================================================

        ranked_indices = np.argsort(final_probabilities)[::-1][:3]

        top3 = []

        for idx in ranked_indices:

            top3.append(
                {
                    "class": FINAL_CLASSES[int(idx)],
                    "probability": float(final_probabilities[idx]),
                }
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": prediction,
            "confidence": confidence,
            "normal_probability": normal_probability,
            "abnormal_probability": abnormal_probability,
            "probabilities": {
                FINAL_CLASSES[i]: float(final_probabilities[i])
                for i in range(len(FINAL_CLASSES))
            },
            "top3": top3,
            "inference_time_ms": elapsed_ms,
            "fusion_weights": fusion_weights,
            "attention_map": attention_map,
            "feature_map": feature_map,
        }
