import torch
import torch.nn as nn

from config import MODEL_CONFIG

from modules.feature_fusion import WeightedAdaptiveFeatureFusion
from modules.attention import FrequencyAwareAttention

from modules.residual_block import ResidualBlock
from modules.se_block import SEBlock

# ==========================================================
# CNN Encoder
# ==========================================================


class CNNEncoder(nn.Module):

    def __init__(self, use_residual=True, use_se=True):

        super().__init__()

        layers = []

        # --------------------------------------------------
        # Block 1
        # --------------------------------------------------

        layers.extend(
            [
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )

        if use_residual:
            layers.append(ResidualBlock(32))

        layers.append(nn.MaxPool2d(2))

        # --------------------------------------------------
        # Block 2
        # --------------------------------------------------

        layers.extend(
            [
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]
        )

        if use_residual:
            layers.append(ResidualBlock(64))

        layers.append(nn.MaxPool2d(2))

        # --------------------------------------------------
        # Block 3
        # --------------------------------------------------

        layers.extend(
            [
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )

        if use_se:
            layers.append(SEBlock(128))

        # Final convolutional layer
        layers.extend(
            [
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(self, x):

        feature_map = None

        for layer in self.features:

            x = layer(x)

            # Capture the final convolutional feature map
            # after the convolution and before global pooling.
            if isinstance(layer, nn.Conv2d) and layer.out_channels == 128:
                feature_map = x

        pooled = x

        feature_vector = pooled.flatten(1)

        return feature_vector, feature_map


# ==========================================================
# Proposed PediaLung-XAI Model
# ==========================================================


class PediaLungXAI(nn.Module):

    def __init__(
        self,
        num_classes=7,
        use_residual=True,
        use_se=True,
        use_fusion=True,
        use_attention=True,
    ):

        super().__init__()

        # --------------------------------------------------
        # Three feature encoders
        # --------------------------------------------------

        self.mfcc_encoder = CNNEncoder(
            use_residual=use_residual,
            use_se=use_se,
        )

        self.mel_encoder = CNNEncoder(
            use_residual=use_residual,
            use_se=use_se,
        )

        self.chroma_encoder = CNNEncoder(
            use_residual=use_residual,
            use_se=use_se,
        )

        # --------------------------------------------------
        # Adaptive Feature Fusion
        # --------------------------------------------------

        self.use_fusion = use_fusion

        if use_fusion:

            self.fusion = WeightedAdaptiveFeatureFusion()

        # --------------------------------------------------
        # Frequency-Aware Attention
        # --------------------------------------------------

        self.use_attention = use_attention

        if use_attention:

            self.attention = FrequencyAwareAttention(feature_dim=128)

        # --------------------------------------------------
        # Lightweight Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        mfcc,
        mel,
        chroma,
    ):

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------

        mfcc_feature, mfcc_feature_map = self.mfcc_encoder(mfcc)

        mel_feature, mel_feature_map = self.mel_encoder(mel)

        chroma_feature, chroma_feature_map = self.chroma_encoder(chroma)

        print("\n========== FEATURE VECTOR STATISTICS ==========")

        print(
            f"MFCC    mean={mfcc_feature.mean().item():.6f} "
            f"std={mfcc_feature.std().item():.6f} "
            f"min={mfcc_feature.min().item():.6f} "
            f"max={mfcc_feature.max().item():.6f}"
        )

        print(
            f"Mel     mean={mel_feature.mean().item():.6f} "
            f"std={mel_feature.std().item():.6f} "
            f"min={mel_feature.min().item():.6f} "
            f"max={mel_feature.max().item():.6f}"
        )

        print(
            f"Chroma  mean={chroma_feature.mean().item():.6f} "
            f"std={chroma_feature.std().item():.6f} "
            f"min={chroma_feature.min().item():.6f} "
            f"max={chroma_feature.max().item():.6f}"
        )

        print("===============================================\n")
        # --------------------------------------------------
        # Feature fusion
        # --------------------------------------------------

        if self.use_fusion:

            fused_feature, fusion_weights = self.fusion(
                mfcc_feature,
                mel_feature,
                chroma_feature,
            )

        else:

            fused_feature = (mfcc_feature + mel_feature + chroma_feature) / 3.0

            fusion_weights = None

        # --------------------------------------------------
        # Frequency-aware attention
        # --------------------------------------------------

        if self.use_attention:

            attention_feature, attention_map = self.attention(fused_feature)

        else:

            attention_feature = fused_feature

            attention_map = None

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        output = self.classifier(attention_feature)

        # --------------------------------------------------
        # Return
        #
        # output
        # fusion_weights
        # attention_map
        # mel_feature_map
        #
        # We keep mel_feature_map as the fourth output
        # because your current Grad-CAM/inference code
        # expects this interface.
        # --------------------------------------------------

        return (
            output,
            fusion_weights,
            attention_map,
            mfcc_feature_map,
        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing PediaLungXAI")
    print("=" * 60)

    # Use the actual proposed architecture.
    model = PediaLungXAI(**MODEL_CONFIG["proposed"])

    # ------------------------------------------------------
    # Dummy inputs
    # ------------------------------------------------------

    mfcc = torch.randn(4, 1, 40, 94)

    mel = torch.randn(4, 1, 128, 259)

    chroma = torch.randn(4, 1, 12, 259)

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    outputs, weights, attention, feature_map = model(
        mfcc,
        mel,
        chroma,
    )

    # ------------------------------------------------------
    # Model information
    # ------------------------------------------------------

    print("\nModel:")
    print(model)

    print("\nOutput shape:")
    print(outputs.shape)

    print("\nFusion weights:")
    print(weights)

    print("\nAttention shape:")

    if attention is not None:
        print(attention.shape)
    else:
        print("Disabled")

    print("\nReturned MFCC feature map:")
    print(feature_map.shape)

    print("\nFeature map statistics:")
    print("Min :", feature_map.min().item())

    print("Max :", feature_map.max().item())

    print("Mean:", feature_map.mean().item())

    print("Std :", feature_map.std().item())

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)
