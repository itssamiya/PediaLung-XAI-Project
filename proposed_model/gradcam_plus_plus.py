import torch
import torch.nn.functional as F


class GradCAMPlusPlus:

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, mfcc, mel, chroma, class_idx):

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        outputs, _, _, feature_map = self.model(
            mfcc,
            mel,
            chroma,
        )

        # Keep gradients of the target feature map
        feature_map.retain_grad()

        # Target class score
        score = outputs[:, class_idx].sum()

        # Backward pass
        score.backward()

        gradients = feature_map.grad
        activations = feature_map

        # Grad-CAM++ weights
        alpha_num = gradients.pow(2)

        alpha_denom = 2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(
            dim=(2, 3),
            keepdim=True,
        )

        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom),
        )

        alpha = alpha_num / alpha_denom

        positive_gradients = F.relu(gradients)

        weights = (alpha * positive_gradients).sum(
            dim=(2, 3),
            keepdim=True,
        )

        # Generate CAM
        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)

        # Use first sample
        cam = cam[0]

        # Normalize
        cam = cam - cam.min()

        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()
