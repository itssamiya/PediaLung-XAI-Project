import torch
import torch.nn as nn


class FrequencyAwareAttention(nn.Module):

    def __init__(self, feature_dim=128):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):

        attention_map = self.attention(x)

        output = x + x * attention_map

        return output, attention_map


if __name__ == "__main__":

    model = FrequencyAwareAttention(feature_dim=128)

    x = torch.randn(8, 128)

    out, att = model(x)

    print("Output :", out.shape)

    print("Attention Shape:", att.shape)
