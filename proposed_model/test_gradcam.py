import torch
import torch.nn.functional as F

from multibranch_model import PediaLungXAI
from gradcam import GradCAM

from config import MODEL_CONFIG

print("=" * 60)
print("GRAD-CAM DIAGNOSTIC TEST")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load model
# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:", device)

model = PediaLungXAI(**MODEL_CONFIG["proposed"]).to(device)

model.eval()


# ----------------------------------------------------------
# 2. Create test inputs
# ----------------------------------------------------------
# These dimensions match your current model test.

mfcc = torch.randn(1, 1, 40, 94, device=device)

mel = torch.randn(1, 1, 128, 259, device=device)

chroma = torch.randn(1, 1, 12, 259, device=device)


# ----------------------------------------------------------
# 3. Initialize Grad-CAM
# ----------------------------------------------------------

print("\nInitializing Grad-CAM...")

gradcam = GradCAM(model)

print("Grad-CAM ready.")


# ----------------------------------------------------------
# 4. Forward pass
# ----------------------------------------------------------

model.zero_grad()

outputs, weights, attention, mel_feature_map = model(
    mfcc,
    mel,
    chroma,
)

predicted_class = outputs.argmax(dim=1).item()

target_score = outputs[0, predicted_class]

print("\nPredicted class:", predicted_class)

print("Target score:", target_score.item())


# ----------------------------------------------------------
# 5. Inspect returned Mel feature map
# ----------------------------------------------------------

print("\n================ MEL FEATURE MAP ================")

print("Shape:", tuple(mel_feature_map.shape))

print("Min:", mel_feature_map.min().item())

print("Max:", mel_feature_map.max().item())

print("Mean:", mel_feature_map.mean().item())

print("Std:", mel_feature_map.std().item())


# ----------------------------------------------------------
# 6. Retain gradient
# ----------------------------------------------------------

mel_feature_map.retain_grad()


# ----------------------------------------------------------
# 7. Backward pass
# ----------------------------------------------------------

target_score.backward()


# ----------------------------------------------------------
# 8. Check gradient
# ----------------------------------------------------------

print("\n================ GRADIENT ================")

if mel_feature_map.grad is None:

    print("ERROR: mel_feature_map.grad is None")

else:

    gradients = mel_feature_map.grad

    print("Shape:", tuple(gradients.shape))

    print("Min:", gradients.min().item())

    print("Max:", gradients.max().item())

    print("Mean:", gradients.mean().item())

    print("Absolute mean:", gradients.abs().mean().item())

    print("Std:", gradients.std().item())


# ----------------------------------------------------------
# 9. Calculate Grad-CAM manually
# ----------------------------------------------------------

if mel_feature_map.grad is not None:

    gradients = mel_feature_map.grad[0]
    activations = mel_feature_map.detach()[0]

    # Global average pooling of gradients

    weights_cam = gradients.mean(dim=(1, 2))

    print("\n================ CAM WEIGHTS ================")

    print("Shape:", tuple(weights_cam.shape))

    print("Min:", weights_cam.min().item())

    print("Max:", weights_cam.max().item())

    print("Mean:", weights_cam.mean().item())

    # Weighted feature maps

    cam = torch.sum(
        weights_cam[:, None, None] * activations,
        dim=0,
    )

    print("\n================ RAW CAM ================")

    print("Shape:", tuple(cam.shape))

    print("Min:", cam.min().item())

    print("Max:", cam.max().item())

    print("Mean:", cam.mean().item())

    # ReLU

    cam = F.relu(cam)

    print("\n================ AFTER RELU ================")

    print("Min:", cam.min().item())

    print("Max:", cam.max().item())

    print("Mean:", cam.mean().item())


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
