import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAMPlusPlus:

    def __init__(self, model):

        self.model = model
        self.model.eval()

        # ==================================================
        # Target layer
        # ==================================================
        #
        # CNNEncoder:
        #
        # 14 = Conv2d(128,128)
        # 15 = BatchNorm2d(128)
        # 16 = ReLU
        # 17 = AdaptiveAvgPool2d
        #
        # Third convolutional feature map of MEL branch
        # Targeting this layer preserves meaningful spatial
        # frequency-time information for Grad-CAM++.
        # the convolution output before BatchNorm is not
        # a suitable visualization target here.
        #
        self.target_layer = self.model.mel_encoder.features[12]

        self.activations = None
        self.gradients = None

        # ==================================================
        # Hooks
        # ==================================================

        self.target_layer.register_forward_hook(self._save_activation)

        self.target_layer.register_full_backward_hook(self._save_gradient)

        print("Grad-CAM++ target layer:")
        print(self.target_layer)

    # ======================================================
    # Forward hook
    # ======================================================

    def _save_activation(
        self,
        module,
        input,
        output,
    ):

        self.activations = output

    # ======================================================
    # Backward hook
    # ======================================================

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    # ======================================================
    # Generate Grad-CAM++
    # ======================================================

    def generate(
        self,
        mfcc,
        mel,
        chroma,
        class_idx,
    ):

        self.model.eval()

        # --------------------------------------------------
        # Clear previous values
        # --------------------------------------------------

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs, _, _, _ = self.model(
            mfcc,
            mel,
            chroma,
        )

        # --------------------------------------------------
        # Target class score
        # --------------------------------------------------

        score = outputs[:, class_idx].sum()

        # --------------------------------------------------
        # Backward pass
        # --------------------------------------------------

        score.backward()

        # --------------------------------------------------
        # Validate hooks
        # --------------------------------------------------

        if self.activations is None:
            raise RuntimeError("Grad-CAM++ ERROR: activation was not captured.")

        if self.gradients is None:
            raise RuntimeError("Grad-CAM++ ERROR: gradient was not captured.")

        activations = self.activations
        gradients = self.gradients

        # ==================================================
        # Diagnostics
        # ==================================================

        print()
        print("=============== GRAD-CAM++ ===============")

        print(
            "Target layer:",
            self.target_layer,
        )

        print(
            "Activation shape:",
            tuple(activations.shape),
        )

        print(
            "Activation min:",
            activations.min().item(),
        )

        print(
            "Activation max:",
            activations.max().item(),
        )

        print(
            "Activation mean:",
            activations.mean().item(),
        )

        print()

        print(
            "Gradient shape:",
            tuple(gradients.shape),
        )

        print(
            "Gradient min:",
            gradients.min().item(),
        )

        print(
            "Gradient max:",
            gradients.max().item(),
        )

        print(
            "Gradient mean:",
            gradients.mean().item(),
        )

        # ==================================================
        # Grad-CAM++ calculation
        # ==================================================

        gradients_2 = gradients.pow(2)

        gradients_3 = gradients.pow(3)

        # --------------------------------------------------
        # Spatial contribution
        # --------------------------------------------------

        spatial_sum = (activations * gradients_3).sum(
            dim=(2, 3),
            keepdim=True,
        )

        # --------------------------------------------------
        # Alpha denominator
        # --------------------------------------------------

        alpha_denom = 2.0 * gradients_2 + spatial_sum

        alpha_denom = torch.where(
            alpha_denom != 0.0,
            alpha_denom,
            torch.ones_like(alpha_denom),
        )

        alpha = gradients_2 / (alpha_denom + 1e-8)

        # --------------------------------------------------
        # Positive gradients
        # --------------------------------------------------

        positive_gradients = F.relu(gradients)

        # --------------------------------------------------
        # Grad-CAM++ channel weights
        # --------------------------------------------------

        weights = (alpha * positive_gradients).sum(
            dim=(2, 3),
            keepdim=True,
        )

        print()
        print("CAM++ weights:")
        print(
            "Min:",
            weights.min().item(),
        )
        print(
            "Max:",
            weights.max().item(),
        )
        print(
            "Mean:",
            weights.mean().item(),
        )

        # ==================================================
        # Generate CAM++
        # ==================================================

        cam = (weights * activations).sum(dim=1)

        # Only positive evidence
        cam = F.relu(cam)

        print()
        print("Raw CAM++:")
        print(
            "Shape:",
            tuple(cam.shape),
        )

        print(
            "Min:",
            cam.min().item(),
        )

        print(
            "Max:",
            cam.max().item(),
        )

        print(
            "Mean:",
            cam.mean().item(),
        )

        # --------------------------------------------------
        # First sample
        # --------------------------------------------------

        cam = cam[0]

        # ==================================================
        # Normalize
        # ==================================================

        cam_min = cam.min()
        cam_max = cam.max()

        variation = (cam_max - cam_min).item()

        if variation < 1e-10:

            print()
            print("WARNING: Grad-CAM++ has almost zero variation.")

            cam = torch.zeros_like(cam)

        else:

            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        print()
        print("Final Grad-CAM++:")

        print(
            "Min:",
            cam.min().item(),
        )

        print(
            "Max:",
            cam.max().item(),
        )

        print(
            "Mean:",
            cam.mean().item(),
        )

        print("==========================================")

        return cam.detach().cpu().numpy()

    # ======================================================
    # End of class
    # ======================================================


# ==========================================================
# Visualization
# ==========================================================


def visualize_gradcam(
    mel,
    heatmap,
    save_path,
    label,
    prediction,
    confidence,
    top3,
):

    # ------------------------------------------------------
    # Convert MEL tensor to numpy
    # ------------------------------------------------------

    if torch.is_tensor(mel):

        mel_image = mel.detach().cpu().numpy()

    else:

        mel_image = np.asarray(mel)

    # Remove batch/channel dimensions
    mel_image = np.squeeze(mel_image)

    # ------------------------------------------------------
    # Resize CAM to MEL dimensions
    # ------------------------------------------------------

    heatmap_tensor = (
        torch.tensor(
            heatmap,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    heatmap_tensor = F.interpolate(
        heatmap_tensor,
        size=mel_image.shape,
        mode="bilinear",
        align_corners=False,
    )

    heatmap_resized = heatmap_tensor[0, 0].numpy()

    # ------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------

    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    # ------------------------------------------------------
    # Create figure
    # ------------------------------------------------------

    plt.figure(figsize=(6, 3), dpi=100)

    # MEL spectrogram
    plt.imshow(
        mel_image,
        aspect="auto",
        origin="lower",
        cmap="gray",
    )

    # Grad-CAM++ overlay
    plt.imshow(
        heatmap_resized,
        aspect="auto",
        origin="lower",
        cmap="jet",
        alpha=0.45,
    )

    plt.colorbar(label="Grad-CAM++ Importance")

    plt.title(
        f"Grad-CAM++ | Prediction: "
        f"{prediction} | Confidence: "
        f"{confidence * 100:.2f}%"
    )

    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=100,
    )

    plt.close()

    print(f"Grad-CAM++ image saved to: " f"{save_path}")

    return save_path


# ==========================================================
# Direct test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GRAD-CAM++ TEST")
    print("=" * 60)

    print("Grad-CAM++ class loaded successfully.")

    print("Target layer will be:")

    print("model.mel_encoder.features[16]")
