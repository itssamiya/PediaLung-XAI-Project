import torch
import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x + x * y
