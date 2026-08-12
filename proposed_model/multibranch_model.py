import torch
import torch.nn as nn

from config import MODEL_CONFIG
from config import EXPERIMENT_NAME


from modules.feature_fusion import WeightedAdaptiveFeatureFusion
from modules.attention import FrequencyAwareAttention

# ==========================================================
# CNN Encoder
# ==========================================================
from modules.residual_block import ResidualBlock
from modules.se_block import SEBlock


class CNNEncoder(nn.Module):

    def __init__(self, use_residual=True, use_se=True):

        super().__init__()

        layers = []

        # ------------------------
        # Block 1
        # ------------------------

        layers.extend(
            [
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ]
        )

        if use_residual:
            layers.append(ResidualBlock(32))

        layers.append(nn.MaxPool2d(2))

        # ------------------------
        # Block 2
        # ------------------------

        layers.extend(
            [
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]
        )

        if use_residual:
            layers.append(ResidualBlock(64))

        layers.append(nn.MaxPool2d(2))

        # ------------------------
        # Block 3
        # ------------------------

        layers.extend(
            [
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )

        if use_se:
            layers.append(SEBlock(128))

        layers.extend(
            [
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ]
        )

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)

    def forward(self, x):

        feature_map = None

        for layer in self.features:

            x = layer(x)

            if not isinstance(layer, nn.AdaptiveAvgPool2d):
                feature_map = x

        pooled = x
        feature_vector = pooled.flatten(1)

        return feature_vector, feature_map


# ==========================================================
# Proposed Model
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

        # Three feature extractors

        self.mfcc_encoder = CNNEncoder(
            use_residual,
            use_se,
        )

        self.mel_encoder = CNNEncoder(
            use_residual,
            use_se,
        )

        self.chroma_encoder = CNNEncoder(
            use_residual,
            use_se,
        )

        # Novel Modules

        if use_fusion:
            self.fusion = WeightedAdaptiveFeatureFusion()

        if use_attention:
            self.attention = FrequencyAwareAttention()

        self.use_fusion = use_fusion
        self.use_attention = use_attention

        # Lightweight classifier

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes)
        )

    def forward(self, mfcc, mel, chroma):

        mfcc_feature, _ = self.mfcc_encoder(mfcc)

        mel_feature, mel_feature_map = self.mel_encoder(mel)

        chroma_feature, _ = self.chroma_encoder(chroma)

        if self.use_fusion:

            fused_feature, weights = self.fusion(
                mfcc_feature,
                mel_feature,
                chroma_feature,
            )

        else:

            fused_feature = (mfcc_feature + mel_feature + chroma_feature) / 3

            weights = None

        if self.use_attention:

            attention_feature, attention_map = self.attention(fused_feature)

        else:

            attention_feature = fused_feature

            attention_map = None

        output = self.classifier(attention_feature)

        return output, weights, attention_map, mel_feature_map


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    model = PediaLungXAI(**MODEL_CONFIG[EXPERIMENT_NAME])

    mfcc = torch.randn(4, 1, 40, 94)

    mel = torch.randn(4, 1, 128, 259)

    chroma = torch.randn(4, 1, 12, 259)

    outputs, weights, attention, feature_map = model(
        mfcc,
        mel,
        chroma,
    )

    print(model)

    print("\nPrediction :", outputs.shape)

    print("Fusion Weights :", weights)

    if attention is not None:
        print("Attention :", attention.shape)
    else:
        print("Attention : Disabled")
