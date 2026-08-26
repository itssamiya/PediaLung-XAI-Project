import torch
import matplotlib.pyplot as plt

from multibranch_model import PediaLungXAI
from config import MODEL_CONFIG, EXPERIMENT_NAME

model = PediaLungXAI(**MODEL_CONFIG[EXPERIMENT_NAME])

checkpoint = torch.load("saved_models/proposed_best.pth", map_location="cpu")

model.load_state_dict(checkpoint)
model.eval()


# ----------------------------------------------------
# Hook
# ----------------------------------------------------

feature_maps = {}


def hook(module, input, output):
    feature_maps["mel"] = output.detach()


layer = model.mel_encoder.features[13]

handle = layer.register_forward_hook(hook)


# ----------------------------------------------------
# Dummy input
# ----------------------------------------------------

mfcc = torch.randn(1, 1, 40, 94)
mel = torch.randn(1, 1, 128, 94)
chroma = torch.randn(1, 1, 12, 94)

with torch.no_grad():
    model(mfcc, mel, chroma)

handle.remove()

maps = feature_maps["mel"][0]

print(maps.shape)

fig, axes = plt.subplots(8, 8, figsize=(12, 12))

for i in range(64):

    ax = axes[i // 8][i % 8]

    ax.imshow(maps[i].cpu(), cmap="jet", aspect="auto")

    ax.axis("off")

plt.tight_layout()

plt.show()
