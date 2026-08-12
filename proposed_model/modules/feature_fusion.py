import torch
import torch.nn as nn


class WeightedAdaptiveFeatureFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.weight_predictor = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, mfcc, mel, chroma):

        combined = torch.cat(
            [mfcc, mel, chroma],
            dim=1,
        )

        weights = self.weight_predictor(combined)

        weights = self.softmax(weights)

        w1 = weights[:, 0].unsqueeze(1)

        w2 = weights[:, 1].unsqueeze(1)

        w3 = weights[:, 2].unsqueeze(1)

        fused = w1 * mfcc + w2 * mel + w3 * chroma

        return fused, weights


if __name__ == "__main__":

    fusion = WeightedAdaptiveFeatureFusion()

    mfcc = torch.randn(8, 128)
    mel = torch.randn(8, 128)
    chroma = torch.randn(8, 128)

    fused, weights = fusion(mfcc, mel, chroma)

    print("Fused Feature :", fused.shape)
    print("Weights :", weights.shape)
    print(weights)
