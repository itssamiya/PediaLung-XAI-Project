import torch
import torch.nn.functional as F

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

# =========================================================
# Load model
# =========================================================

device = torch.device("cpu")

model = PediaLungXAI(
    num_classes=2,
    **cfg.MODEL_CONFIG[cfg.MODEL_NAME],
).to(device)

model.load_state_dict(
    torch.load(
        "saved_models/hierarchical_binary_best.pth",
        map_location=device,
    )
)

model.eval()


# =========================================================
# Test audio
# =========================================================

wav_path = r"D:\PediaLung-XAI\data\test2022_wav" r"\40512331_8.1_1_p1_3544.wav"


# =========================================================
# Preprocessing
# =========================================================

signal, sr = load_audio(wav_path)

signal = wavelet_denoise(signal)

signal = normalize_length(signal, sr)

mfcc = extract_mfcc(signal, sr)
mel = extract_mel(signal, sr)
chroma = extract_chroma(signal, sr)


mfcc = (
    torch.tensor(
        mfcc,
        dtype=torch.float32,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

mel = (
    torch.tensor(
        mel,
        dtype=torch.float32,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

chroma = (
    torch.tensor(
        chroma,
        dtype=torch.float32,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)


# =========================================================
# Check input statistics
# =========================================================

print("\n================ INPUT STATISTICS ================")

for name, x in [
    ("MFCC", mfcc),
    ("Mel", mel),
    ("Chroma", chroma),
]:

    print(
        f"{name:8s} | "
        f"min={x.min().item():.6e} | "
        f"max={x.max().item():.6e} | "
        f"mean={x.mean().item():.6e} | "
        f"std={x.std().item():.6e}"
    )


# =========================================================
# Save intermediate activations
# =========================================================

activations = {}
gradients = {}


def forward_hook(module, inputs, output):

    activations["target"] = output.detach()


def backward_hook(module, grad_input, grad_output):

    gradients["target"] = grad_output[0].detach()


target_layer = model.mel_encoder.features[14]

print("\nTarget layer:")
print(target_layer)

target_layer.register_forward_hook(forward_hook)

target_layer.register_full_backward_hook(backward_hook)


# =========================================================
# Forward
# =========================================================

output, weights, attention, mel_feature_map = model(
    mfcc,
    mel,
    chroma,
)


print("\n================ MODEL OUTPUT ================")

print("Output:", output)

print(
    "Probabilities:",
    F.softmax(output, dim=1),
)

print(
    "Fusion weights:",
    weights,
)

print(
    "Attention mean:",
    attention.mean().item() if attention is not None else None,
)


# =========================================================
# Returned Mel feature map
# =========================================================

print("\n================ MEL FEATURE MAP ================")

print(
    "Shape:",
    mel_feature_map.shape,
)

print(
    "Min:",
    mel_feature_map.min().item(),
)

print(
    "Max:",
    mel_feature_map.max().item(),
)

print(
    "Mean:",
    mel_feature_map.mean().item(),
)

print(
    "Std:",
    mel_feature_map.std().item(),
)


# =========================================================
# Hook activation
# =========================================================

print("\n================ HOOK ACTIVATION ================")

if "target" in activations:

    a = activations["target"]

    print(
        "Shape:",
        a.shape,
    )

    print(
        "Min:",
        a.min().item(),
    )

    print(
        "Max:",
        a.max().item(),
    )

    print(
        "Mean:",
        a.mean().item(),
    )

    print(
        "Std:",
        a.std().item(),
    )

else:

    print("NO ACTIVATION CAPTURED")


# =========================================================
# Backward
# =========================================================

model.zero_grad()

predicted_class = output.argmax(dim=1).item()

score = output[:, predicted_class]

print(
    "\nPredicted class:",
    predicted_class,
)

print(
    "Target score:",
    score.item(),
)

score.backward()


# =========================================================
# Gradient statistics
# =========================================================

print("\n================ GRADIENT ================")

if "target" in gradients:

    g = gradients["target"]

    print(
        "Shape:",
        g.shape,
    )

    print(
        "Min:",
        g.min().item(),
    )

    print(
        "Max:",
        g.max().item(),
    )

    print(
        "Mean:",
        g.mean().item(),
    )

    print(
        "Absolute mean:",
        g.abs().mean().item(),
    )

    print(
        "Std:",
        g.std().item(),
    )

else:

    print("NO GRADIENT CAPTURED")


print("\n=================================================")
print("DIAGNOSTIC COMPLETE")
print("=================================================")
