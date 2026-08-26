import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config as cfg

from multibranch_model import PediaLungXAI

from utils.preprocessing import (
    load_audio,
    wavelet_denoise,
    normalize_length,
    extract_mfcc,
    extract_mel,
    extract_chroma,
)


class HierarchicalPredictor:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using Device:", self.device)

        # =====================================================
        # Label encoders
        # =====================================================

        df = pd.read_csv("features/labels.csv")

        # Original 7-class encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df["label"])

        print("\n7-Class Labels:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(i, ":", label)

        # =====================================================
        # Binary model
        # =====================================================

        print("\nLoading Binary Model...")

        self.binary_model = PediaLungXAI(
            num_classes=2,
            **cfg.MODEL_CONFIG[cfg.MODEL_NAME],
        ).to(self.device)

        self.binary_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.MODEL_DIR,
                    "hierarchical_binary_best.pth",
                ),
                map_location=self.device,
            )
        )

        self.binary_model.eval()

        print("Binary model loaded.")

        # =====================================================
        # Abnormal model
        # =====================================================

        print("\nLoading Abnormal Model...")

        self.abnormal_model = PediaLungXAI(
            num_classes=6,
            **cfg.MODEL_CONFIG[cfg.MODEL_NAME],
        ).to(self.device)

        self.abnormal_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.MODEL_DIR,
                    "hierarchical_abnormal_best.pth",
                ),
                map_location=self.device,
            )
        )

        self.abnormal_model.eval()

        print("Abnormal model loaded.")

        # =====================================================
        # Abnormal class mapping
        # =====================================================

        self.abnormal_classes = [
            "Coarse Crackle",
            "Fine Crackle",
            "Rhonchi",
            "Stridor",
            "Wheeze",
            "Wheeze+Crackle",
        ]

        print("\nAbnormal Classes:")

        for i, name in enumerate(self.abnormal_classes):
            print(i, ":", name)

    # =========================================================
    # Preprocessing
    # =========================================================

    def preprocess(self, wav_path):

        signal, sr = load_audio(wav_path)

        signal = wavelet_denoise(signal)

        signal = normalize_length(signal, sr)

        mfcc = extract_mfcc(signal, sr)

        mel = extract_mel(signal, sr)

        chroma = extract_chroma(signal, sr)

        return mfcc, mel, chroma

    # =========================================================
    # Tensor preparation
    # =========================================================

    def prepare_tensors(self, mfcc, mel, chroma):

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        chroma = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return (
            mfcc.to(self.device),
            mel.to(self.device),
            chroma.to(self.device),
        )

    # =========================================================
    # Hierarchical prediction
    # =========================================================

    def predict(self, wav_path):

        mfcc, mel, chroma = self.preprocess(wav_path)

        mfcc, mel, chroma = self.prepare_tensors(
            mfcc,
            mel,
            chroma,
        )

        # =====================================================
        # Stage 1: Normal vs Abnormal
        # =====================================================

        with torch.no_grad():

            binary_outputs, _, _, _ = self.binary_model(
                mfcc,
                mel,
                chroma,
            )

            binary_probs = F.softmax(
                binary_outputs,
                dim=1,
            )

            binary_confidence, binary_prediction = torch.max(
                binary_probs,
                dim=1,
            )

        # Binary encoding:
        #
        # 0 = Normal
        # 1 = Abnormal
        #

        binary_class = binary_prediction.item()

        # =====================================================
        # Normal
        # =====================================================

        if binary_class == 0:

            prediction = "Normal"

            confidence = binary_confidence.item()

            top3 = [
                ("Normal", float(binary_probs[0, 0])),
                ("Abnormal", float(binary_probs[0, 1])),
            ]

            return (
                prediction,
                confidence,
                top3,
                "Normal",
                binary_confidence.item(),
            )

        # =====================================================
        # Abnormal
        # =====================================================

        with torch.no_grad():

            abnormal_outputs, _, _, _ = self.abnormal_model(
                mfcc,
                mel,
                chroma,
            )

            abnormal_probs = F.softmax(
                abnormal_outputs,
                dim=1,
            )

            top_probs, top_indices = torch.topk(
                abnormal_probs,
                k=3,
                dim=1,
            )

        predicted_abnormal = top_indices[0, 0].item()

        prediction = self.abnormal_classes[predicted_abnormal]

        confidence = top_probs[0, 0].item()

        top3 = []

        for p, idx in zip(
            top_probs[0],
            top_indices[0],
        ):

            class_name = self.abnormal_classes[idx.item()]

            top3.append(
                (
                    class_name,
                    float(p.item()),
                )
            )

        return (
            prediction,
            confidence,
            top3,
            "Abnormal",
            binary_confidence.item(),
        )


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    predictor = HierarchicalPredictor()

    test_file = r"D:\PediaLung-XAI\data\test2022_wav" r"\40512331_8.1_1_p1_3544.wav"

    (
        prediction,
        confidence,
        top3,
        binary_result,
        binary_confidence,
    ) = predictor.predict(test_file)

    print("\n========================================")
    print("HIERARCHICAL PREDICTION")
    print("========================================")

    print("Binary Decision :", binary_result)
    print(f"Binary Confidence : " f"{binary_confidence * 100:.2f}%")

    print("Final Prediction :", prediction)

    print(f"Final Confidence : " f"{confidence * 100:.2f}%")

    print("\nTop-3 Predictions:")

    for name, prob in top3:

        print(f"{name:20s} " f"{prob * 100:.2f}%")
