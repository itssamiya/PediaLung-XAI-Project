import gc
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import time
import config as cfg

from sklearn.preprocessing import LabelEncoder

from multibranch_model import PediaLungXAI
from gradcam import GradCAM, visualize_gradcam

from utils.preprocessing import (
    load_audio,
    wavelet_denoise,
    normalize_length,
    extract_mfcc,
    extract_mel,
    extract_chroma,
)

from config import SAVE_DIR
from config import MODEL_DIR
from config import EXPERIMENT_NAME

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.models import efficientnet_b0


class LungSoundPredictor:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model...")

        self.model = PediaLungXAI(
            num_classes=7,
            **cfg.MODEL_CONFIG[cfg.MODEL_NAME],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    MODEL_DIR,
                    f"{EXPERIMENT_NAME}_best.pth",
                ),
                map_location=self.device,
            )
        )

        self.model.eval()

        print("Model loaded.")

        self.label_encoder = LabelEncoder()

        df = pd.read_csv("features/labels.csv")

        self.label_encoder.fit(df["label"])

        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model)

    #######################################################
    # Preprocessing
    #######################################################


def preprocess(self, wav_path):

    print(">>> PREPROCESS ENTERED <<<", flush=True)

    signal, sr = load_audio(wav_path)

    print(">>> AUDIO LOADED <<<", flush=True)

    signal = wavelet_denoise(signal)

    signal = normalize_length(signal, sr)

    mfcc = extract_mfcc(signal, sr)

    mel = extract_mel(signal, sr)

    print(">>> MEL EXTRACTED <<<", flush=True)
    print("Shape:", mel.shape, flush=True)
    print("Min:", mel.min(), flush=True)
    print("Max:", mel.max(), flush=True)
    print("Mean:", mel.mean(), flush=True)
    print("Std:", mel.std(), flush=True)

    chroma = extract_chroma(signal, sr)

    return mfcc, mel, chroma

    #######################################################
    # Tensor Conversion
    #######################################################

    def prepare_tensors(self, mfcc, mel, chroma):

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        chroma = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return (mfcc.to(self.device), mel.to(self.device), chroma.to(self.device))

    #######################################################
    # Grad-CAM
    #######################################################

    def generate_gradcam(
        self, mfcc, mel, chroma, predicted_class, confidence, top_predictions
    ):

        heatmap = self.gradcam.generate(mfcc, mel, chroma, predicted_class)

        os.makedirs("results", exist_ok=True)

        save_path = "results/latest_gradcam.png"

        prediction_name = self.label_encoder.classes_[predicted_class]

        visualize_gradcam(
            mfcc,
            heatmap,
            save_path,
            "Unknown",
            prediction_name,
            confidence,
            top_predictions,
        )
        return save_path

    #######################################################
    # Prediction
    #######################################################

    def predict(self, wav_path):

        print(">>> PREDICT FUNCTION RUNNING <<<")

        total_start = time.perf_counter()

        pre_start = time.perf_counter()

        mfcc, mel, chroma = self.preprocess(wav_path)

        mfcc, mel, chroma = self.prepare_tensors(
            mfcc,
            mel,
            chroma,
        )

        total_time = (time.perf_counter() - total_start) * 1000

        pre_time = (time.perf_counter() - pre_start) * 1000

        infer_start = time.perf_counter()

        with torch.no_grad():

            outputs, _, _, _ = self.model(
                mfcc,
                mel,
                chroma,
            )

            infer_time = (time.perf_counter() - infer_start) * 1000

            probs = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

            # Top-3 predictions
            top_probs, top_indices = torch.topk(
                probs,
                k=3,
                dim=1,
            )

            top_predictions = []

            for p, idx in zip(top_probs[0], top_indices[0]):

                class_name = self.label_encoder.classes_[idx.item()]

                top_predictions.append(
                    (
                        class_name,
                        float(p.item()),
                    )
                )

        prediction = self.label_encoder.classes_[predicted.item()]

        grad_start = time.perf_counter()

        gradcam_path = self.generate_gradcam(
            mfcc,
            mel,
            chroma,
            predicted.item(),
            confidence.item(),
            top_predictions,
        )

        grad_time = (time.perf_counter() - grad_start) * 1000

        elapsed = (time.perf_counter() - total_start) * 1000

        return (
            prediction,
            confidence.item(),
            gradcam_path,
            top_predictions,
            elapsed,
            pre_time,
            infer_time,
            grad_time,
        )


#######################################################
# Test
#######################################################

if __name__ == "__main__":

    predictor = LungSoundPredictor()

    (
        prediction,
        confidence,
        image,
        top3,
        elapsed,
        pre_time,
        infer_time,
        grad_time,
    ) = predictor.predict(
        r"D:\PediaLung-XAI\data\test2022_wav\40512331_8.1_1_p1_3544.wav"
    )

    print("\nPrediction :", prediction)
    print(f"Confidence          : {confidence*100:.2f}%")

    print("\nPerformance")
    print(f"Preprocessing Time : {pre_time:.2f} ms")
    print(f"Model Inference    : {infer_time:.2f} ms")
    print(f"Grad-CAM Time      : {grad_time:.2f} ms")
    print(f"Total Pipeline     : {elapsed:.2f} ms")

    print("\nTop-3 Predictions")
    for name, prob in top3:
        print(f"{name:20s} {prob*100:.2f}%")

    print("\nGrad-CAM Image :", image)
