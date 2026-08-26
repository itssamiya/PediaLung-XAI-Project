 PediaLung-XAI: Explainable Pediatric Respiratory Sound Classification

**PediaLung-XAI** is an interpretable, lightweight deep learning diagnostic framework designed for pediatric respiratory sound classification using the **SPRSound** dataset. The system combines stationary Wavelet Denoising, 3-channel spectral feature fusion (MFCC + Mel-Spectrogram + Chromagram), and an EfficientNet-B0 backbone with dynamic Grad-CAM visual explainability.

---

 Key Features

* **Acoustic Preprocessing**: Stationary Wavelet Denoising to filter ambient noise while preserving acoustic transients.
* **3-Channel Feature Fusion**: Synchronized concatenation of MFCCs, Mel-Spectrograms, and Chromagrams into spatial tensors (128x259).
* **Lightweight Backbone**: EfficientNet-B0 (~4.02M parameters) achieving **85.44% Accuracy** and **85.28% Weighted F1** score.
* **Explainable AI (XAI)**: Dual-panel Grad-CAM visualization mapping class activation heatmaps over input Mel-Spectrograms.
* **Desktop GUI Application**: CustomTkinter interface supporting file uploads, real-time diagnostic classification, top-3 class likelihoods, and latency performance benchmarks.

---

## 📊 Model Performance Comparison

| Model Architecture | Accuracy | Weighted F1 | Parameter Count |
| :--- | :---: | :---: | :---: |
| **EfficientNet-B0 (Final Model)** | **85.44%** | **85.28%** | **4.02 M** |
| Lightweight CNN  | 82.73% | 82.75% | **4.02 M** |
| ResNet-18 | 75.15% | 77.66% | 11.18 M |

---

##  Repository Directory Structure

```text
PediaLung-XAI-Project/
│
├── proposed_model/
│   ├── gui.py                  # Primary CustomTkinter Graphical User Interface
│   ├── app_inference.py        # Core EfficientNet-B0 Inference & Grad-CAM Engine
│   ├── config.py               # Hyperparameters and path configurations
│   └── comparison_models/      # Baseline comparative training & benchmark scripts
│
├── utils/
│   └── preprocessing.py        # Audio loading, wavelet denoising, & feature extraction
│
├── archive/                    # Legacy experimental models & diagnostic test scripts
├── features/                   # Class label metadata
└── README.md                   # Project documentation
